# 开发者：李永桢
# 开发时间：2024/8/7 9:43 PM
# 代码功能：传播记录带标签的五元组；传播过程中新事件的引入；

"""
原始图数据：
假设已经有一张有特征向量、标签、邻接矩阵的图；
图上的节点我们视作用户，现在要在这个用户网络上做事件的传递，其中事件是一个和节点特征向量同纬度的向量。

事件传递：
首先随机生成一个事件向量，然后随机选择一个节点作为事件的传播起点，这个节点会向其所有邻居发送这个事件向量，
邻居有一定的概率被激活，这个概率是这个邻居节点的度的倒数，被激活的邻居会接收这个事件的影响，具体的影响方式是会改变这个邻居的特征向量，

特征向量 = 原始的特征向量 + 传播事件的节点的传播系数 * 传播事件的节点的特征向量 + 影响系数 * 事件向量，
这个影响系数和传播系数的范围是0.01～0.05，为每个节点随机生成一个影响系数和传播系数保存在图的结构里面。

然后被激活的邻居也会继续传播这个事件给它的所有邻居，以此类推，
事件终止的条件是这个事件传播到的节点中没有一个节点被激活。
"""

import networkx as nx
import torch
import numpy as np
import random
import threading
import scipy.sparse as sp
from torch_geometric.utils import from_networkx
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
# from threading import Lock
# from queue import Queue
# import time

def set_random_seeds(seed=42):
    """
    设置随机种子

    :param seed: 随机种子
    :return: 没有返回值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def initialize_graph(data_path, num_nodes_to_select, k, n):
    """
    初始化图数据，包括读取数据；创建子图；转换为PyTorch Geometric图；生成节点特征和标签；生成节点属性传播系数和事件影响系数；

    参数:
    data_path -- 数据文件路径。
    num_nodes_to_select -- 随机选择的节点数。
    k -- 类别数量。
    n -- 特征维度。

    返回:
    G_pyg_subgraph -- 初始化后的PyTorch Geometric图对象。
    """
    '''1. 从文件中读取边数据'''
    edges = []
    with open(data_path, 'r') as file:
        for line in file:
            node1, node2 = map(int, line.strip().split())
            edges.append((node1, node2))

    '''2. 随机选择部分节点和对应的边，构建子图'''
    # # random.seed(42)
    # # range(max(max(edges))) 生成一个从 0 到最大节点值的整数序列，random.sample 从这个序列中随机选择节点
    # # 如果生成的子图不是连通的，则可能会丢失一些节点，例如随机选取 500 个节点，实际：Graph with 427 nodes and 1405 edges
    # selected_nodes = random.sample(range(max(max(edges))), num_nodes_to_select)
    # # 这个列表推导式遍历 edges 列表中的每一条边，只保留那些两个节点都在 selected_nodes 中的边
    # selected_edges = [edge for edge in edges if edge[0] in selected_nodes and edge[1] in selected_nodes]

    '''2. 全图'''
    # 创建 NetworkX 图
    G_nx = nx.Graph()
    G_nx.add_edges_from(edges)

    # 将 NetworkX 图转换为 PyTorch Geometric 图：PyG 在将 NetworkX 图转换为 PyG 图时会自动对节点重新编号
    G_pyg = from_networkx(G_nx)

    # 节点个数
    num_nodes = G_pyg.num_nodes

    '''3. 随机生成类中心和节点特征'''
    # class_centers 存储了 k 个类的中心点，每个中心点是一个 n 维向量
    class_centers = np.random.rand(k, n)

    # labels 存储了每个节点的类别标签，表示每个节点属于哪个类，其中的值是从 0 到 k-1 之间的随机整数
    labels = np.random.randint(0, k, size=num_nodes)

    # features 用于存储每个节点的特征向量，初始值全部为 0
    features = np.zeros((num_nodes, n))

    for i in range(num_nodes):
        # 循环遍历每个节点，获取当前节点所属类的中心点向量
        class_center = class_centers[labels[i]]
        # 生成噪声：一个大小为 n 的一维数组，其中的值是服从均值为 0，标准差为 0.1 的正态分布的随机数
        noise = np.random.normal(0, 0.1, size=n)
        # 当前节点的特征向量 = 类中心点 + 噪声向量
        features[i] = class_center + noise

    G_pyg.x = torch.tensor(features, dtype=torch.float)
    G_pyg.y = torch.tensor(labels, dtype=torch.long)

    '''4. 随机生成 传播系数 和 事件影响系数，并将其作为节点属性保存'''
    # 传播系数：在传播事件的时候，节点会传播自己的个人立场，这个值代表传播个人立场的多少
    propagation_coefficients = np.random.uniform(0.01, 0.05, size=num_nodes)
    # 事件影响系数：在传播事件的时候，节点会对事件进行修改，这里简单的设置为一个影响系数代表夸大或者缩小事件
    event_influence_coefficients = np.random.uniform(0.1, 0.2, size=num_nodes)

    G_pyg.propagation = torch.tensor(propagation_coefficients, dtype=torch.float)
    G_pyg.event_influence = torch.tensor(event_influence_coefficients, dtype=torch.float)

    return G_pyg


def generate_events(G, num_events):
    """
    生成多个事件，每个事件包含事件向量、起始节点和优先级。
    这里事件是有可能从同一个节点产生的，因为没有设置起始节点的唯一性

    参数:
    G -- PyTorch Geometric 图对象。
    num_events -- 要生成的事件数量。

    返回:
    events -- 事件列表。
    """
    events = []
    for i in range(num_events):
        # 生成随机的事件向量、起始节点和优先级
        event_vector = np.random.rand(G.x.shape[1])
        start_node = np.random.randint(G.num_nodes)
        priority = i + 1
        event_id = f'event_{i + 1}'  # 为每个事件生成唯一的ID

        events.append({
            'event_vector': event_vector,
            'start_node': start_node,
            'priority': priority,
            'event_id': event_id
        })
        print(f"Event {i + 1}: Start node = {start_node}, "
              f"Event vector = {event_vector}, Priority = {priority}, ID = {event_id}")
    print("--- Event Initialization Finished ---")
    print()

    return events


def calculate_node_degree(node, edge_index):
    """
    计算节点的度。

    参数:
    node -- 要计算度的节点。
    edge_index -- 图的边索引。

    返回:
    degree -- 节点的度。
    """
    ''' edge_index[0] == node: 这部分代码创建了一个布尔掩码，用于选择 edge_index 第一行中所有等于 node 的元素。
        掩码的长度与边的数量相同，对于每条边，如果源节点是 node，则对应的掩码值为 True，否则为 False。'''
    ''' edge_index[1][...]: 这部分代码使用上面创建的布尔掩码来选择 edge_index 第二行中的相应元素。
        结果是一个一维张量，包含了所有与 node 相连的目标节点的编号。'''
    '''.numpy(): 这是对张量调用的方法，用于将 PyTorch 张量转换为 NumPy 数组。'''
    neighbors = edge_index[1][edge_index[0] == node].numpy()
    degree = len(neighbors)
    return degree


def propagate_events(G, events, max_new_events):
    """
    处理多个事件的传播过程。

    参数:
    G -- PyTorch Geometric 图对象。
    events -- 事件列表。
    max_new_events -- 最大新事件数。

    返回:
    propagation_log -- 传播记录的列表。
    """
    threads = []                # 存储所有线程的列表
    lock = threading.Lock()     # 线程锁，用于线程安全地访问共享资源
    propagation_log = []        # 记录传播过程的列表：五元组
    processed = {}              # 记录已经处理过的事件
    global_timestep = 1         # 全局时间步，从1开始
    new_events_count = 0        # 记录新引入的事件数

    """
    新事件的生成和引入：
        new_event 函数负责生成新事件，包括事件向量（随机生成）、起始节点（随机选择）和优先级（事件列表长度+1）。
    生成新事件后，将其加入事件列表，并启动一个新的线程来传播该事件。
    """
    def new_event():
        '''
        :return:
        '''

        ''' 声明 new_events_count 引用的是外层的变量，也就是说在 new_event 函数中
            修改 propagate_events 函数作用域中的 new_events_count 变量。'''
        nonlocal new_events_count
        new_event_vector = np.random.rand(G.x.shape[1])     # 事件向量
        new_start_node = np.random.randint(G.num_nodes)     # 事件起始节点
        new_priority = len(events) + 1                      # 事件优先级：事件列表长度+1
        new_event_id = f'event_{new_priority}'              # 事件ID

        new_event = {
            'event_vector': new_event_vector,
            'start_node': new_start_node,
            'priority': new_priority,
            'event_id': new_event_id
        }

        events.append(new_event)    # 将新生成事件添加到事件列表
        new_events_count += 1       # 新生成事件+1
        print()
        print(f"New Event {new_priority}: Start node = {new_start_node}, "
              f"Event vector = {new_event_vector}, Priority = {new_priority}, ID = {new_event_id}")
        print()

        '''创建一个新的线程 new_thread，线程的目标函数是 propagate_event，并传递了一组参数'''
        ''' G：图对象。
            new_event：新生成的事件。
            new_event['event_id']：事件的唯一标识符。
            processed：记录已经处理过的事件的字典。
            lock：线程锁，用于保证线程安全。
            propagation_log：记录事件传播过程的列表。
            global_timestep：全局时间步。'''
        new_thread = threading.Thread(target=propagate_event,
                                      args=(G, new_event, new_event['event_id'], processed,
                                            lock, propagation_log, global_timestep))
        '''用于启动一个新线程，使其开始执行目标函数 propagate_event'''
        new_thread.start()
        '''将启动的线程添加到 threads 列表中，以便后续可以管理和监控这些线程'''
        threads.append(new_thread)

    '''为 events 列表中的每个事件创建了一个新线程，线程的目标函数是 propagate_event'''
    for event in events:
        # event['event_id'] = f"event_{event['priority']}"
        thread = threading.Thread(target=propagate_event,
                                  args=(G, event, event['event_id'], processed, lock,
                                        propagation_log, global_timestep))
        thread.start()
        threads.append(thread)

    while any(thread.is_alive() for thread in threads) or new_events_count < max_new_events:
        # # 等待所有线程完成当前时间步
        # for thread in threads:
        #     thread.join()

        # 移除已完成的线程
        threads = [t for t in threads if t.is_alive()]

        # 增加全局时间步
        global_timestep += 1

        # 在每个时间步后检查并引入新事件
        if new_events_count < max_new_events and random.random() < 0.3:  # 30% 的概率引入新事件
            new_event()

        # 等待一个时间步
        threading.Event().wait(1)  # 等待一秒钟以模拟时间步

    return propagation_log


def propagate_event(G, event, event_id, processed, lock, propagation_log, global_timestep):
    """
    单个事件的传播过程。

    参数:
    G -- PyTorch Geometric 图对象。
    event -- 事件字典。
    event_id -- 事件的编号。
    processed -- 已处理节点的集合。
    lock -- 线程锁对象。
    propagation_log -- 传播记录的列表。
    global_timestep -- 全局时间步计数。
    """
    start_node = event['start_node']       # 事件传播的起始节点：生成事件的时候确定
    event_vector = event['event_vector']   # 事件向量
    priority = event['priority']           # 事件优先级
    affected_nodes = []                    # 用于收集被当前事件影响的节点

    def activate_node(node):
        """
        判断节点 node 是否被激活；
        判断条件：节点按照 1/度 的概率被激活，即节点的度数越高，激活的概率就越低。生成一个随机数，如果随机数小于激活概率，则激活该节点。
        参数 node: 判断的节点
        返回值: True / False 表示是否激活
        """
        # 节点的度
        degree = calculate_node_degree(node, G.edge_index)

        # 激活概率被设置为节点度数的倒数。即节点的度数越高，激活的概率就越低。
        activation_prob = 1.0 / degree

        # 生成一个随机数，如果随机数小于激活概率，则激活该节点
        return random.random() < activation_prob

    # 初始化一个队列，这个队列用于广度优先搜索（BFS），帮助追踪事件从起始节点向外传播的过程，
    queue = [(start_node, global_timestep)]
    # 初始化一个集合（local_processed），该集合用于记录在当前传播过程中已经处理过的节点，以避免重复处理同一个节点。
    local_processed = set([start_node])

    while queue:
        current_node, timestep = queue.pop(0)                     # 当前节点，当前节点被处理的时间步
        current_feature = G.x[current_node].numpy()               # 当前节点的特征
        propagation = G.propagation[current_node].item()          # 当前节点的传播系数：在传播事件的时候，如何传播自己的个人立场
        event_influence = G.event_influence[current_node].item()  # 当前节点的事件影响系数：在传播事件的时候，如何修改事件

        ''' 更新节点特征向量 '''
        # 获取当前节点 current_node 的所有邻居节点
        neighbors = list(G.edge_index[1, G.edge_index[0] == current_node].numpy())

        # 遍历当前节点 current_node 的所有邻居节点
        for neighbor in neighbors:

            # 如果邻居节点既未被处理过，又被成功激活，则将其添加到 local_processed 集合中
            if neighbor not in local_processed and activate_node(neighbor):
                local_processed.add(neighbor)

                # 持有锁：在多线程环境中，同一时刻只有一个线程可以进入这个代码块，从而避免数据竞争和不一致的情况
                with lock:

                    # 1.检查邻居节点是否已经在全局处理过的节点集合 processed 中。如果不在，则需要处理该节点
                    # 2.如果邻居节点已经在 processed 中，则检查其当前处理的事件优先级。如果当前事件的优先级（priority）比记录中的优先级更高，则需要更新该节点的处理信息。
                    # if neighbor not in processed or processed[neighbor][1] > priority:
                    if neighbor not in processed:

                        # 将邻居节点的处理信息更新到 processed 字典中；键：节点，值：(影响节点的事件ID，优先级)
                        processed[neighbor] = (event_id, priority)

                        # 将邻居节点 neighbor 和更新后的时间步 timestep + 1 添加到处理队列 queue 中
                        # 这意味着在下一时间步将处理该邻居节点，继续事件的传播
                        queue.append((neighbor, timestep + 1))

                        # 在被当前事件影响的节点列表中添加被影响的邻居节点
                        affected_nodes.append(neighbor)

                        # 经过 当前节点 修改加工后传播的 事件向量
                        event_feature = propagation * current_feature + event_influence * event_vector

                        # 将当前事件的传播记录添加到 propagation_log 列表中
                        propagation_log.append((current_node, neighbor, event_feature, timestep, event_id))

                        # 更新邻居特征向量
                        # new_feature = G.x[neighbor].numpy() + event_feature
                        # G.x[neighbor] = torch.tensor(new_feature, dtype=torch.float)
                        G.x[neighbor] = G.x[neighbor] + event_feature

    affected_nodes_count = len(affected_nodes)  # 统计被当前事件影响的节点的总数
    print(f"Event {event_id}: Start node = {start_node}, Priority = {priority}, "
          f"Affected nodes count = {affected_nodes_count}, Affected nodes = {affected_nodes}")


def perform_clustering(G):
    """
    使用K-Means算法对图的节点特征进行聚类，并计算聚类结果的准确率。

    参数:
    G -- PyTorch Geometric 图对象。

    返回:
    None -- 该函数直接打印聚类结果和准确率。
    """

    k = len(set(G.y.numpy()))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(G.x.numpy())
    predicted_labels = kmeans.labels_

    accuracy = accuracy_score(G.y.numpy(), predicted_labels)
    print(f"Clustering accuracy: {accuracy}")
    print("--- Clustering Finished ---")
    print()

    return predicted_labels


def convert_matrix_format(G):
    """
    将特征矩阵和邻接矩阵转换为COO格式稀疏矩阵。

    参数:
    G -- PyTorch Geometric 图对象。

    返回:
    features_coo -- COO格式的特征矩阵。
    adj_coo -- COO格式的邻接矩阵。
    """
    features = G.x.numpy()
    edge_index = G.edge_index.numpy()

    num_nodes = G.num_nodes
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        adj_matrix[edge_index[0, i], edge_index[1, i]] = 1

    features_coo = sp.coo_matrix(features)
    adj_coo = sp.coo_matrix(adj_matrix)

    return features_coo, adj_coo


# 主程序
if __name__ == "__main__":
    '''1. 初始化图数据'''
    # set_random_seeds(seed=42)
    data_path = "/Users/liyongzhen/Desktop/Postgraduate/GraphClustering/code/chushihua/data/facebook_combined.txt"
    G = initialize_graph(data_path=data_path, num_nodes_to_select=500, k=10, n=5)

    '''2. 生成事件'''
    events = generate_events(G, num_events=5)
    print()

    '''3. 事件传播前进行聚类并打印准确率'''
    print("Clustering before event propagation:")
    perform_clustering(G)

    '''4. 传播事件'''
    propagation_log = propagate_events(G, events, max_new_events=3)
    count = 0
    print()
    for i in propagation_log:
        count += 1
        print(i)
    print("count=", count)

    '''5. 事件传播后进行聚类并打印准确率'''
    print()
    print("Clustering after event propagation:")
    perform_clustering(G)

    '''6. 转换特征矩阵和邻接矩阵为COO格式稀疏矩阵'''
    print(f"G:{G}")
    # features_coo, adj_coo = convert_matrix_format(G)
    #
    # print(f"Features (COO format):\n{features_coo}")
    # print(f"Adjacency matrix (COO format):\n{adj_coo}")

    '''7. 将节点特征、节点标签、边集写入txt文件'''
    # 将节点特征写入txt文件
    with open('data/event_propagation.txt', 'w') as f:
        for i in range(G.num_nodes):
            feature_str = ' '.join(map(str, G.x[i].tolist()))
            f.write(feature_str + '\n')

    # 将节点标签写入txt文件
    with open('data/event_propagation_label.txt', 'w') as f:
        for label in G.y.tolist():
            f.write(str(label) + '\n')

    # 将边集写入txt文件
    with open('graph/event_propagation_graph.txt', 'w') as f:
        for i in range(G.edge_index.shape[1]):  # 遍历每一条边
            start_node = G.edge_index[0, i].item()  # 起始节点
            end_node = G.edge_index[1, i].item()  # 到达节点
            f.write(f'{start_node} {end_node}\n')  # 写入文件，节点之间用空格分隔

