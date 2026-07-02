# 开发者：李永桢
# 开发时间：2024/8/7 8:28 PM
# 代码功能：生成 KNN（K近邻） 图

import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import normalize


def construct_graph(f_name, features, label, method='heat', topK=10):
    """

    :参数 features: 特征数据，np二维数组
    :参数 label:    标签数据，np数组
    :参数 method:   相似度矩阵的计算方法，可选的有：[heat 热核|cos 余弦相似度|ncos 归一化的余弦相似度]
    :返回值:
    """

    '''0. 准备的变量'''
    num = len(label)  # 节点个数
    dist = None       # 距离（相似度）

    '''1. 计算相似度（距离）矩阵'''
    if method == 'heat':
        '''
        1.method == 'heat'：计算相似度矩阵的方法是热核
        2.pdist(...)：用于计算特征矩阵 features 中所有点对之间的距离。
          参数 'sqeuclidean' 指定了距离度量是平方欧几里得距离，即每个点对的距离是它们特征向量差的平方和。
          返回的是一个一维数组，其中包含了所有点对之间的距离，但不是以矩阵的形式。
        3.squareform(...)：将这个一维数组转换成一个对称的二维距离矩阵，这样每个样本之间的距离就可以通过矩阵中的元素直接访问。
        '''
        t = 0.5  # 热核相似度的参数
        dist = -t * squareform(pdist(features, 'sqeuclidean'))
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = features @ features.T
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = features @ features.T

    '''2. 为特征矩阵中的每个样本找到其最近的 topK 个邻居的索引'''
    inds = []                       # 初始化一个空列表，用于存储每个样本的最近邻居的索引
    for i in range(dist.shape[0]):  # dist.shape[0]：样本的行数，即样本数量
        '''
        1.dist[i, :]: 获取距离矩阵的第i行，这表示样本 i 与其他所有样本之间的距离。

        2.np.argpartition(..., -(topK+1)): 使用 argpartition 函数找到距离最小的 topK+1 个元素的索引。
          这里 -(topK+1) 是参数，表示我们想要找到最小的 topK+1 个值。

        3.切片操作的语法是[start:stop:step]，其中 start 是可选的，默认是从 0 开始；
          stop 是必需的，表示切片的结束位置（不包括这个位置）；step 也是可选的，默认是1。

        3.[-(topK+1):]: 这是一个切片操作，它从找到的索引中选取最后 topK 个，
          这是因为 argpartition 返回的索引是未排序的，而最后一个索引是样本i与自己的距离，我们不需要它。
        '''
        ind = np.argpartition(dist[i, :], -(topK + 1))[-(topK + 1):]
        inds.append(ind)  # 将找到的最近邻居的索引列表添加到 inds 列表中

    '''3. 构建 KNN 图：把边集写入文件'''
    with open(f_name, 'w') as f:      # 写入文件
        counter = 0                   # 初始化一个计数器 counter，用于计算图中的错误率
        for i, v in enumerate(inds):  # i 是索引，v 是当前元素的值，即与当前样本 i 距离最近的 topK 个样本的索引列表
            for vv in v:              # 对每个索引列表 v 中的每个索引 vv 进行循环
                if vv == i:           # 如果等于当前样本的索引，那么跳过这个循环，因为我们不需要与自己连接
                    continue
                if label[vv] != label[i]:  # 如果 vv 的标签与 i 指向的样本的标签不同，那么这表示它们属于不同的类别，将 counter +1
                    counter += 1
                f.write(f'{i} {vv}\n')     # 将当前样本i和它的最近邻居vv之间的连接写入文件

    # 打印出错误率，即 counter 除以 num * topK 的结果。这表示在构建的 KNN 图中，不同类别之间的边所占的比例
    print(f'error rate: {counter / (num * topK)}')


"""
    执行函数
"""


'''1. 创建命令行解析器'''
parser = argparse.ArgumentParser(
    description='train',  # 描述，当命令行工具使用--h或--help选项时，这个描述会显示在帮助信息的开始部分，告诉用户这个脚本的用途
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 一个帮助消息格式化类，它会为每个参数显示其默认值
# 添加超参数
parser.add_argument('--name', type=str, default='facebook')     # 数据集名字
parser.add_argument('--k', type=int, default=3)                 # k 近邻的取值
# 解析超参数
args = parser.parse_args()

'''1. 读取样本特征数据、标签数据，转换为 np 数组（因为距离函数需要这种格式）'''
features_filename = 'data/{}.txt'.format(args.name)         # 保存特征矩阵的文件名
labels_filename = 'data/{}_label.txt'.format(args.name)     # 保存标签的文件名
features = np.genfromtxt(features_filename, delimiter=' ')  # 转换为 np 二维数组的特征数据
labels = np.genfromtxt(labels_filename, dtype=int)          # 转换为 np 二维数组的标签数据

'''2. 调用函数 construct_graph，生成 KNN 图'''
# 输出文件，这个文件将会包含根据给定特征和标签构建的KNN（k-最近邻）图的邻接信息
f_name = 'graph/{}_graph_knn.txt'.format(args.name)
construct_graph(f_name=f_name, features=features, label=labels, method='heat', topK=3)

