# 开发者：李永桢
# 开发时间：2024/10/14 3:43 PM
# 代码功能：计算节点的二阶相似性，只计算二阶邻居

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

'''1. 计算节点 u 和节点 v 之间的非对称余弦相似度'''
def asymmetric_cosine_similarity(A_csr, u, v):
    # 获取节点 u 和 v 的邻接向量
    a_u = A_csr.getrow(u).toarray().flatten()  # 将稀疏行转换为密集行
    a_v = A_csr.getrow(v).toarray().flatten()

    # 计算余弦相似度
    dot_product = np.dot(a_u, a_v)
    norm_u = np.linalg.norm(a_u)
    norm_v = np.linalg.norm(a_v)

    if norm_u == 0 or norm_v == 0:
        return 0  # 如果任意一个向量的范数为 0，返回 0
    return dot_product / (norm_u * norm_v)

'''2. 计算图的二阶相似度矩阵 S，仅计算二阶邻居的相似度'''
def compute_second_order_similarity(A_csr):
    num_nodes = A_csr.shape[0]
    S = sp.lil_matrix((num_nodes, num_nodes))  # 使用 LIL 格式来构建稀疏矩阵，适合逐元素修改

    # 遍历每个节点的二阶邻居，并计算相似度
    for u in tqdm(range(num_nodes), desc="Calculating second-order similarities"):
        # 获取节点 u 的一阶邻居
        neighbors_1st = A_csr.getrow(u).indices

        # 遍历一阶邻居的邻居（即二阶邻居）
        for v in neighbors_1st:
            neighbors_2nd = A_csr.getrow(v).indices
            for w in neighbors_2nd:
                if w != u:  # 跳过自环
                    S[u, w] = asymmetric_cosine_similarity(A_csr, u, w)

    return S.tocsr()  # 转换为 CSR 格式返回

'''3. 计算输入矩阵 A_S'''
def compute_input_matrix(A, S, eta):
    return A + eta * S

'''4. 应用阈值，超过阈值的元素设为 1'''
def apply_threshold(A_S, threshold):
    return sp.csr_matrix((A_S >= threshold).astype(int))

'''5. 从 txt 文件中读取邻接矩阵并返回稀疏矩阵格式（COO），然后转换为 CSR 格式'''
def read_sparse_adj_matrix_from_txt(file_path, num_nodes):
    """
    从 txt 文件中读取邻接矩阵并返回稀疏矩阵格式（COO），然后转换为 CSR 格式。

    Parameters:
    file_path -- txt 文件路径
    num_nodes -- 节点数

    Returns:
    A_csr -- 稀疏邻接矩阵（CSR格式）
    """
    rows = []
    cols = []

    with open(file_path, 'r') as file:
        for line in file:
            u, v = map(int, line.strip().split())
            rows.append(u)
            cols.append(v)
            # 由于文件中已经包含对称边，因此不需要再添加对称边
            # rows.append(v)  # 对称图（无向图）
            # cols.append(u)  # 添加对称边

    # 构造稀疏邻接矩阵（COO格式）
    data = np.ones(len(rows))
    A_coo = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

    # 转换为CSR格式返回
    return A_coo.tocsr()

'''6. 将邻接矩阵写入 txt 文件，每行表示一条边'''
def write_adj_matrix_to_txt(A, file_path):
    with open(file_path, 'w') as file:
        coo = A.tocoo()  # 转换为 COO 格式以便遍历非零元素
        for u, v in zip(coo.row, coo.col):
            file.write(f"{u} {v}\n")

'''7. 统计新增加的边数'''
def count_new_edges(A_old, A_new):
    return np.sum((A_new > A_old).astype(int))


'''8. 添加边阈值的选择：百分位数阈值'''
'''选择一个百分位数（如90%、95%），将相似度矩阵中高于该百分位数的相似度视为高相关性并添加边。这种方法确保仅选择最强的节点相似性。'''
def determine_threshold_percentile(S, percentile=95):
    # 直接提取稀疏矩阵中的非零值
    non_zero_values = S.data  # 取出稀疏矩阵的非零元素
    # 选择指定百分位数的相似度值作为阈值
    threshold = np.percentile(non_zero_values, percentile)
    return threshold


'''8. 添加边阈值的选择：基于均值和标准差的阈值'''
'''使用相似度矩阵的均值和标准差来选择一个动态阈值，通常可以设定为均值加上若干倍的标准差（如 𝜇+2𝜎）作为阈值。'''
def determine_threshold_mean_std(S, k=2):
    """
    基于均值和标准差动态选择阈值。

    Parameters:
    S -- 二阶相似度矩阵（CSR格式）
    k -- 控制标准差的倍数

    Returns:
    threshold -- 动态选择的阈值
    """
    # 直接提取稀疏矩阵的非零元素
    non_zero_values = S.data

    # 计算均值和标准差
    mean = np.mean(non_zero_values)
    std = np.std(non_zero_values)

    # 动态计算阈值
    threshold = mean + k * std
    return threshold


'''8. 添加边阈值的选择：固定比例边数'''
'''根据想要添加的边的比例选择阈值，比如只保留前10%或前5%的高相似度边。'''
def determine_threshold_top_k(S, top_k_percent=10):
    """
    基于 top-k percent 选择相似度阈值。

    Parameters:
    S -- 二阶相似度矩阵（CSR格式）
    top_k_percent -- 前k%的阈值比例

    Returns:
    threshold -- 动态选择的阈值
    """
    # 提取稀疏矩阵中的非零值
    non_zero_values = S.data

    # 排除零相似度并排序
    non_zero_values = non_zero_values[non_zero_values != 0]
    non_zero_values.sort()

    # 根据 top_k_percent 选择阈值
    top_k_index = int(len(non_zero_values) * (1 - top_k_percent / 100))
    threshold = non_zero_values[top_k_index]
    return threshold


'''主函数：计算输入矩阵，并应用改进'''
if __name__ == "__main__":

    '''2. 超参数 eta'''
    eta = 0.5

    '''3. 读取邻接矩阵'''
    # 真实数据集
    dataset = 'dblp'                                                 # 数据集名字
    data = np.loadtxt('data/{}.txt'.format(dataset))                 # 特征矩阵
    num_nodes, _ = data.shape                                        # 节点数

    # # 自创数据集
    # dataset = 'a'  # 数据集名字
    # num_nodes = 20

    # 图数据文件路径
    input_file_path = "graph/{}_graph.txt".format(dataset)           # 输入图数据文件路径
    output_file_path = "./graph/{}_graph_enhance.txt".format(dataset)  # 输出新图数据文件路径

    A = read_sparse_adj_matrix_from_txt(input_file_path, num_nodes)  # 生成稀疏邻接矩阵

    '''4. 计算二阶相似度矩阵'''
    '''ACM'''
    '''DBLP：选择阈值方式一，percentile=95，新增加的边数: 2092'''
    '''DBLP：选择阈值方式二，k=3，新增加的边数: 636'''
    '''DBLP：选择阈值方式三，top_k_percent=1，新增加的边数: 624（当前）'''
    S = compute_second_order_similarity(A)
    # threshold = determine_threshold_percentile(S, percentile=95)      # 选择阈值方式一
    # threshold = determine_threshold_mean_std(S, k=3)                # 选择阈值方式二
    threshold = determine_threshold_top_k(S, top_k_percent=1)       # 选择阈值方式三
    threshold = threshold * eta


    '''5. 计算输入矩阵 A_S'''
    A_S = compute_input_matrix(A, S, eta)

    '''6. 应用阈值来增加边'''
    A_new = apply_threshold(A_S, threshold)

    '''7. 统计新增加的边'''
    new_edges_count = count_new_edges(A, A_new)

    '''8. 输出新邻接矩阵'''
    write_adj_matrix_to_txt(A_new, output_file_path)
    print(f"新增加的边数: {new_edges_count}")
    print(f"处理后的邻接矩阵 A_new:\n{A_new}")
