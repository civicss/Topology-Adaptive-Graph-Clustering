# 开发者：李永桢
# 开发时间：2024/9/6 2:27 PM
# 代码功能：Kmeans 聚类算法

import numpy as np
import torch


'''1. 初始化聚类中心'''
def initialize(X, num_clusters):
    """
    参数：
    X：一个形状为(num_samples, num_features)的PyTorch张量，其中num_samples是数据集中的样本数量，num_features是每个样本的特征数量。
    num_clusters：一个整数，表示要初始化的簇的数量。

    返回值：
    initial_state：一个形状为(num_clusters, num_features)的NumPy数组，这个数组存储了每个簇的初始中心。
    """
    num_samples = len(X)        # 数据集中样本的数量

    # 随机选取初始聚类中心的索引：用 NumPy 的 random.choice 函数随机选择 num_clusters 个不重复的样本索引。
    indices = np.random.choice(num_samples, num_clusters, replace=False)

    # 获得初始聚类中心的矩阵的：使用这些索引从X张量中提取样本，得到初始状态。这些样本将被用作每个簇的初始中心
    initial_state = X[indices]

    # 返回一个包含初始簇中心的NumPy数组
    return initial_state


'''2. K-means 算法'''
def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    参数：
    X：特征矩阵，一个形状为(num_samples, num_features)的 PyTorch 张量
    num_clusters：簇的数量
    distance：表示用于计算样本之间距离的度量方法，可以是’euclidean’（欧几里得距离）或’cosine’（余弦距离）
    tol：一个浮点数，表示收敛的阈值。如果连续迭代中中心点的移动小于这个阈值，算法停止
    device：一个PyTorch设备对象，用于指定数据和模型应该在CPU还是GPU上运行

    返回值：
    choice_cluster.cpu()：一个形状为(num_samples,)的PyTorch张量，其中每个元素是一个整数，表示每个样本所属的簇
    dis.cpu()：一个形状为(num_samples, num_clusters)的PyTorch张量，其中每个元素是一个浮点数，表示每个样本到每个簇中心的距离
    initial_state.cpu()：一个形状为(num_clusters, num_features)的PyTorch张量，其中每个元素是一个浮点数，表示每个簇中心的特征值
    """
    # print(f'running k-means on {device}..')
    '''1. 根据参数 distance 决定距离度量方法'''
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    '''2. 将输入数据 X 转换为浮点类型，并移动到指定的 device'''
    X = X.float()
    X = X.to(device)

    '''3. 初始化聚类中心：通过 initialize 函数实现，它随机选择数据集中的样本作为初始中心
          这段代码尝试通过多次随机初始化聚类中心，并选择一个使得所有样本到其最近中心的总距离最小的初始化方法'''

    # dis_min是一个变量，用于存储 所有样本 到其最近中心的总距离的最小值，初始时，将其设置为无穷大（float('inf')）。
    dis_min = float('inf')
    # initial_state_best是一个变量，用于存储最佳的初始化聚类中心的集合，初始时，将其设置为None，表示还没有找到最佳的初始化方法。
    initial_state_best = None
    # 尝试 20 次初始化
    for i in range(20):
        initial_state = initialize(X, num_clusters)               # 调用 initialize 函数，获得 num_clusters 个初始聚类中心
        dis = pairwise_distance_function(X, initial_state).sum()  # 计算所有样本到其最近中心的总距离
        if dis < dis_min:                                         # 如果当前总距离更小，那么当前聚类中心就更优
            dis_min = dis
            initial_state_best = initial_state

    '''4. K-means 核心迭代部分'''
    initial_state = initial_state_best   # 20 次最佳初始化聚类中心
    iteration = 0                        # 初始化迭代次数变量 iteration 为0
    while True:
        # 计算每个样本到当前聚类中心的距离矩阵dis
        dis = pairwise_distance_function(X, initial_state)

        # 为每个样本找到最近的簇中心：返回一个形状为(num_samples,)的PyTorch张量，其中每个元素是一个整数，表示每个样本所属的簇。
        choice_cluster = torch.argmin(dis, dim=1)

        # 创建 initial_state 的一个副本 initial_state_pre，用于存储更新前的聚类中心
        initial_state_pre = initial_state.clone()

        # 遍历每个簇中心
        for index in range(num_clusters):
            # 找到属于当前簇的样本索引：
            # torch.nonzero(...) 返回一个形状为 (num_samples,) 的 PyTorch 张量，每个元素是一个布尔值，表示是否属于当前簇
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)   # 使用 selected 中的索引从 X 中提取属于当前簇的样本
            initial_state[index] = selected.mean(dim=0)     # 更新当前簇中心为提取样本的平均值

        # 计算聚类中心移动的量：计算每个样本中心与前一次迭代中的中心之间的平方距离，然后开平方得到欧几里得距离，然后求和
        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # 迭代次数 +1
        iteration = iteration + 1

        # 迭代次数超过 500 次，或者中心移动的量的平方小于给定的阈值 tol，则退出循环
        if iteration > 500:
            break
        if center_shift ** 2 < tol:
            break

    return choice_cluster.cpu(), dis.cpu(), initial_state.cpu()


'''3. 聚类'''
def clustering(feature, true_labels, cluster_num):
    """"""
    '''1. 使用 K-means 进行聚类'''
    predict_labels, dis, initial = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean",
                                          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    '''2. 评估聚类结果'''
    acc, nmi, ari, f1 = eva(true_labels, predict_labels.numpy(), show_details=False)

    '''3. 返回：聚类算法预测的标签，形状为(num_samples,)；距离矩阵，形状为(num_samples, num_clusters)；'''
    return 100 * acc, 100 * nmi, 100 * ari, 100 * f1, predict_labels.numpy(), dis



def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    # print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2,
                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # 转换为 tensor，如果是 numpy 数组
    if isinstance(data1, np.ndarray):
        data1 = torch.from_numpy(data1)
    if isinstance(data2, np.ndarray):
        data2 = torch.from_numpy(data2)

    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


