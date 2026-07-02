# 开发者：李永桢
# 开发时间：2024/7/31 6:19 PM
# 代码功能：
"""
1.加入进度条；2.加快计算；3.只匹配 1 种 motif
"""

import torch
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


'''1. 将Scipy稀疏矩阵转换为PyTorch稀疏张量的函数'''
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

'''2. 加载图数据并返回邻接矩阵和节点的邻居列表：不对节点重新编号'''
def load_graph(dataset, k=None, enhance=False):
    """"""
    '''1. 边文件路径：首先判断是否是 KNN 图，再判断是否是增强图'''
    if k:
        if enhance:
            print(f'是使用的相似性增强图,k={k}')
            path = 'graph/{}{}_graph_enhance.txt'.format(dataset, k)
        else:
            print(f'是使用的原始图,k={k}')
            path = 'graph/{}{}_graph.txt'.format(dataset, k)
    elif enhance:
        print("是使用的相似性增强图")
        path = 'graph/{}_graph_enhance.txt'.format(dataset)
    else:
        print("是使用的原始图")
        path = 'graph/{}_graph.txt'.format(dataset)

    '''2. 特征矩阵文件路径'''
    data = np.loadtxt('data/{}.txt'.format(dataset))   # 特征矩阵
    n, _ = data.shape                                  # 节点数

    '''3. 读取边并转换为邻接矩阵'''
    idx = np.array([i for i in range(n)], dtype=np.int32)       # 节点索引
    idx_map = {node: index for index, node in enumerate(idx)}   # 节点映射：原始编号:新的编号
    '''
    从指定路径读取边信息，存储为 Numpy 数组
    edges_unordered = np.array([[101, 203],
                            [101, 304],
                            [203, 405],
                            [304, 506],
                            [405, 506]], dtype=np.int32)
    '''
    edges_unordered = np.genfromtxt(path, dtype=np.int32)

    '''
    .flatten()：把二维数组转换成一维数组：[101, 203, 101, 304, 203, 405, 304, 506, 405, 506]；
    例如，idx_map = {101: 0, 203: 1, 304: 2, 405: 3, 506: 4}
    map 函数：会对 .flatten() 中的每个元素应用 idx_map.get 函数，得到一个迭代器：[0, 1, 0, 2, 1, 3, 2, 4, 3, 4]；
    list 函数：使用 list 函数将 map 返回的迭代器转换为列表：[0, 1, 0, 2, 1, 3, 2, 4, 3, 4]
    np.array(...)：将这个列表转换为一个数据类型为32位整数 NumPy 数组：np.array([0, 1, 0, 2, 1, 3, 2, 4, 3, 4], dtype=np.int32)
    reshape(...)：将一维数组重新塑形为与 edges_unordered 相同的形状。
                edges_unordered.shape 返回原始二维数组的形状 (5, 2)，所以最终结果是：
                edges = np.array([[0, 1],
                              [0, 2],
                              [1, 3],
                              [2, 4],
                              [3, 4]], dtype=np.int32)
    '''
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 创建邻接矩阵，稀疏矩阵格式
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)

    # 构建对称的邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    '''4. 创建节点的邻居列表'''
    adjacent_vertices = []  # 所有顶点的邻接顶点表：一个列表，列表的每个元素是一个列表

    # 初始化 adjacent_vertices 列表
    for vertex_index in range(n):
        adjacent_list_for_vertex = []                       # 创建一个空列表来存储与当前顶点相邻的顶点
        adjacent_vertices.append(adjacent_list_for_vertex)  # 把这个列表加入到所有节点的顶点表中

    # 遍历 edges 列表来填充 adjacent_vertices
    for edge_index in range(len(edges)):
        a, b = edges[edge_index]        # 获取当前边的两个顶点

        adjacent_vertices[a].append(b)  # 将顶点 b 添加到顶点 a 的邻接列表中
        adjacent_vertices[b].append(a)  # 将顶点 a 添加到顶点 b 的邻接列表中（双向边）

    # 将 Scipy 稀疏矩阵转换为 PyTorch 稀疏张量
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, adjacent_vertices


'''3. 加载图数据并返回邻接矩阵和节点的邻居列表：对节点重新编号'''
def load_graph_reindex(dataset, k=None):
    """
    加载图数据并返回邻接矩阵和节点的邻居列表。

    参数:
    - dataset (str): 图的数据集名称。
    - k (int): 可能的motif大小，如果k不为None，则添加到数据集名称中。

    返回:
    - adj (torch.sparse.FloatTensor): 邻接矩阵。
    - adjacent_vertices (list): 每个节点的邻居列表。
    """

    '''1. 边文件路径'''
    if k:
        path = 'graph/{}{}_graph.txt'.format(dataset, k)
    else:
        path = 'graph/{}_graph.txt'.format(dataset)

    '''2. 特征矩阵文件路径'''
    data = np.loadtxt('data/{}.txt'.format(dataset))   # 特征矩阵
    n, _ = data.shape                                  # 节点数

    '''3. 读取边并转换为邻接矩阵'''
    '''
    从指定路径读取边信息，存储为 Numpy 数组
    edges_unordered = np.array([[101, 203],
                            [101, 304],
                            [203, 405],
                            [304, 506],
                            [405, 506]], dtype=np.int32)
    '''
    edges_unordered = np.genfromtxt(path, dtype=np.int32)

    '''获取所有唯一节点 ID，np.unique 默认是排序的'''
    unique_nodes = np.unique(edges_unordered.flatten())

    '''创建从原始节点 ID 到新索引的映射'''
    idx_map = {node: idx for idx, node in enumerate(unique_nodes)}

    '''
    .flatten()：把二维数组转换成一维数组：[101, 203, 101, 304, 203, 405, 304, 506, 405, 506]；
    例如，idx_map = {101: 0, 203: 1, 304: 2, 405: 3, 506: 4}
    map 函数：会对 .flatten() 中的每个元素应用 idx_map.get 函数，得到一个迭代器：[0, 1, 0, 2, 1, 3, 2, 4, 3, 4]；
    list 函数：使用 list 函数将 map 返回的迭代器转换为列表：[0, 1, 0, 2, 1, 3, 2, 4, 3, 4]
    np.array(...)：将这个列表转换为一个数据类型为32位整数 NumPy 数组：np.array([0, 1, 0, 2, 1, 3, 2, 4, 3, 4], dtype=np.int32)
    reshape(...)：将一维数组重新塑形为与 edges_unordered 相同的形状。
                edges_unordered.shape 返回原始二维数组的形状 (5, 2)，所以最终结果是：
                edges = np.array([[0, 1],
                              [0, 2],
                              [1, 3],
                              [2, 4],
                              [3, 4]], dtype=np.int32)
    '''
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 创建邻接矩阵，稀疏矩阵格式
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)

    # 构建对称的邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    '''4. 创建节点的邻居列表'''
    adjacent_vertices = []  # 所有顶点的邻接顶点表：一个列表，列表的每个元素是一个列表

    # 初始化 adjacent_vertices 列表
    for vertex_index in range(n):
        adjacent_list_for_vertex = []                       # 创建一个空列表来存储与当前顶点相邻的顶点
        adjacent_vertices.append(adjacent_list_for_vertex)  # 把这个列表加入到所有节点的顶点表中

    # 遍历 edges 列表来填充 adjacent_vertices
    for edge_index in range(len(edges)):
        a, b = edges[edge_index]        # 获取当前边的两个顶点

        adjacent_vertices[a].append(b)  # 将顶点 b 添加到顶点 a 的邻接列表中
        adjacent_vertices[b].append(a)  # 将顶点 a 添加到顶点 b 的邻接列表中（双向边）

    # 将 Scipy 稀疏矩阵转换为 PyTorch 稀疏张量
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, adjacent_vertices


"""上面内容是加载原始图数据"""
"""下面内容是生成增强图数据"""


'''
1. 匹配 motif：
   参数：m，是一个 motif，比如[2, 2, 2]，表示一个三角形，其中每个节点的度 = 2
'''
def motif_match(m):
    """"""
    '''1. motif m 中的节点数量，比如三角形就是 3'''
    num = len(m)

    '''2. 遍历图中每个可能的起始节点，直到剩余的节点数量不足以形成 motif'''
    for i in tqdm(range(0, size - num + 1), desc="Matching Motifs"):
        '''
        1. 对于每个节点，创建两个集合
           cur_sub 用来存储从这个节点出发，当前匹配的子图节点，初始化包含当前节点 i；
           v_ext 用来存储可扩展的节点，即当前节点的邻居节点，初始化为空；
        '''
        cur_sub = set()
        cur_sub.add(i)
        v_ext = set()

        '''2. 递归地检查从当前节点 i 开始的子图与 motif m 是否匹配'''
        f1(m, cur_sub, v_ext, i, num)

        '''3. 回溯：从 cur_sub 集合移除起始节点，下一次迭代从 i+1 节点开始搜索'''
        cur_sub.remove(i)


'''
2. 递归地在图中搜索与给定的 motif m 匹配的子图
   参数：m--motif；  cur_sub--当前的子图；  v_ext--可选节点：与当前节点相邻的节点；  i--当前节点；  num--motif B 中节点个数
'''
def f1(m, cur_sub, v_ext, i, num):
    """"""
    '''1. 递归出口：
          检查当前子图 cur_sub 是否已经包含了 num 个节点，即与 motif m 的大小相同。
          如果是，则调用 f2 函数来比较当前子图的度与 motif B 的度，并更新匹配结果矩阵 F。'''
    if len(cur_sub) == num:
        f2(m, cur_sub, num)
        return

    '''2. 获得当前节点 i 的可扩展节点个数：len3'''
    v_ext_copy = v_ext.copy()   # 创建 v_ext 的副本 v_ext_copy
    len2 = int(len(N[i]) / sr)  # 用当前节点 i 的邻居列表长度 N[i] 除以 sr，得到阈值 len2，用于控制节点扩展范围
    len3 = max(len2, 0)         # len3 决定了递归过程中 v_ext 集合可以扩展的最大数量，即如果C[i]长度 < es，则len3将是0，不能扩展

    '''3. 获得当前节点 i 的可扩展节点：遍历 0~len3-1 所有可能的邻居节点，确保 v_ext 集合最多可以扩展到 len3 个节点'''
    for j in range(0, len3):
        # 检查当前邻居节点的编号 C[i][j] 是否大于起始节点 i，如果是，则 C[i][j] 是一个有效的邻居节点，可以添加到 v_ext 集合中
        if N[i][j] > i:
            v_ext.add(N[i][j])
    # 将集合转变为列表，因为集合是可变的，列表是不可变的
    v_ext_list = list(v_ext)

    '''4. 扩展子图：遍历节点 i 的可选节点，所有编号大于 i 的邻居节点'''
    for j in v_ext_list:
        # 检查当前邻居节点的编号 jj 是否大于起始节点 i，如果是，则 jj 是一个有效的邻居节点，可以添加到 cur_sub 集合（当前子图）中
        if j > i:
            cur_sub.add(j)                   # 添加到 cur_sub（当前子图）中
            v_ext.remove(j)                  # 因为已经添加到当前子图中，所以从可选节点集中移除
            f1(m, cur_sub, v_ext, j, num)    # 递归：从 j 开始作为新的起始节点开始搜素
            cur_sub.remove(j)                # 回溯：j 已经考虑过，考虑这个位置能够放 i 的其他邻居
            v_ext.add(j)                     # 回溯：恢复原来的 v_ext

    '''5. 在递归过程中，每次从某个节点 i 出发时，都会向 v_ext 集合中添加一些节点（即那些邻居节点），以便在进一步的递归中使用。
          然而，在回溯过程中，我们需要确保 v_ext 集合的状态与之前一致，避免在后续递归层次中重复考虑已经不再相关的节点。'''
    for j in range(0, len3):
        if N[i][j] > i:
            if N[i][j] not in v_ext_copy:
                if N[i][j] in v_ext:
                    v_ext.remove(N[i][j])


'''3. 匹配 motif 及其变体：计算当前子图的度，然后与 motif 度进行比较，如果匹配，则更新 F'''
def f2(m, cur_sub, num):
    """"""
    '''1. 初始化'''
    degree_cs = [0] * num        # 初始化一个长度为 num 的列表 degree_cs，用于存储当前子图中每个节点的度
    index_cs = [0] * num         # 初始化一个长度为 num 的列表 index_cs，用于存储当前子图中每个节点的索引
    count = 0                    # 初始化计数器 count，用于遍历当前子图的节点

    '''2. 计算当前子图的度，记住当前子图每个节点的索引'''
    for i in cur_sub:         # 遍历当前子图 cur_sub 中的每个节点 i，将节点 i 的索引存储在 D[count] 中
        index_cs[count] = i
        for j in cur_sub:     # 计算子图 cur_sub 中，节点 i 的度
            if i != j:
                if A[i][j] == 1:
                    degree_cs[count] += 1
        count += 1

    '''3. 筛选出度 >= motif 的节点的度的节点'''
    max2 = []                       # 初始化一个空列表 max2，用于存储那些度大于等于 B[0] 的节点
    max2.append(index_cs[0])        # 将 index_cs[0]（第一个节点的索引）添加到 max2 中

    # 对 m 进行降序排序（motif 的度序列）
    m.sort(reverse=True)

    # 筛选出度大于等于 B[0] 的节点
    for i in range(1, num):
        if degree_cs[i] >= m[0]:
            max2.append(index_cs[i])

    # 对当前子图的度序列进行降序排序
    degree_cs.sort(reverse=True)

    '''4. 比较 motif 的度m 和 子图的度degree_cs，如果子图匹配 motif，则更新 F'''
    if cmp(m, degree_cs, num):
        # print(f"匹配上{len(m)}边形，匹配的节点是：{index_cs}，当前子图排序后的度：{degree_cs}")
        for i in max2:
            for j in max2:
                if i != j:
                    M[i][j] += motif_significance
                    M[j][i] += motif_significance


'''4. 比较子图是否和 motif m 匹配'''
def cmp(m, degree_cs, num):
    for i in range(num):
        if m[i] > degree_cs[i]:
            return False
    return True






""" 
方法调用：匹配得到的矩阵F，存储到 {dataset}_graph_motif.txt 文件中；F[i][j]指代节点 i 和 j 依据 motif 所产生的关系程度
"""
'''
1. 全局变量
    name: 当前数据集的名称。用于加载对应的数据集文件。
    sr: 从一个点出发找子图，选择邻居的比例，如果是5就是选择1/5
    f_name: 保存匹配结果的文件名。
    A: 邻接矩阵
    C: 每个节点的邻居列表，表示图中节点的相邻节点。
    size: 图中节点的数量。
    M: 匹配结果矩阵，用于存储节点之间的 motif 匹配关系。
    motif_significance: 某种motif的重要程度
'''
'''
需要改的地方: 1.数据集名字; 2.邻居节点比例; 3.k 近邻数量; 4.是否使用增强图;
'''
global name, sr, f_name, A, N, size, M, motif_significance

'''2. 数据集定义'''
# DataSet = ['acm', 'cite', 'dblp', 'usps', 'reut', 'hhar', 'event_propagation', 'cora']    # 7 个数据集
# SelectionRate = [5, 2, 2, 1, 1, 2]                           # 匹配时，选择的一定的数量比例，5就是除以5
# K = [None, None, None, 3, 3, 5]                              # 不同数据集 K 近邻的取值

DataSet = ['usps']          # 1.数据集
SelectionRate = [30]        # 2.匹配时，选择的一定的数量比例的邻居节点，5就是除以5，就是选择1/5邻居
K = [10]                    # 3.k 近邻数量


'''3. 匹配 motif'''
for i in range(0, 1):   # 循环处理多个数据集
    '''1.初始化：每次要改 motif_num'''
    name = DataSet[i]                               # 当前数据集名称
    sr = SelectionRate[i]                           # 从一个点出发找子图，选择邻居的比例，如果是 5 就是选择 1/5
    A, N = load_graph(name, k=K[i], enhance=False)   # 4.加载原始图数据（不重新编号节点，即可能不连续）：邻接矩阵A 和邻居列表N
    size = A.size(0)                                # 图中节点的数量

    '''2. 定义motif：数字代表节点的度'''
    motif_list = [[2, 2, 2],             # A-三角形
                  [2, 2, 2, 2],          # B-四边形
                  [2, 2, 2, 2, 2]]       # C-五边形

    '''3. 匹配过程：motif_significance 指代某种 motif 的重要程度'''
    motif_types = ['triangle', 'quadrilateral', 'pentagon']     # motif 类型
    motif_sign = [5, 4, 3]                                      # motif 每条边权重
    motif_num = [3, 4, 5]                                       # motif 的形状
    motif_list = [motif_list[0], motif_list[1], motif_list[2]]

    for i, motif in enumerate(motif_list):
        M = torch.zeros(size, size)                                        # 初始化匹配结果矩阵：全0
        motif_significance = motif_sign[i]                                 # 设置motif_significance
        motif_match(motif)                                                 # 匹配motif
        f_name = 'graph/{}_graph_motif_{}.txt'.format(name, motif_num[i])  # 保存匹配结果的文件
        np.savetxt(f_name, M.numpy(), fmt='%1.0f')
        print(f'---匹配{motif_types[i]} motif 结束---\n')



