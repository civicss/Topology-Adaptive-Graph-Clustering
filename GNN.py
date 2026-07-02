# 开发者：李永桢
# 开发时间：2024/8/4 10:14 PM
# 代码功能：GCN 层的实现

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()

        # 输入特征维度，输出特征维度
        self.in_features = in_features
        self.out_features = out_features

        # 权重矩阵：大小为 [in_features, out_features]，用 xavier_uniform_ 方法进行初始化，有助于网络快速收敛
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        # support = 输入特征矩阵 features * 权重矩阵 self.weight；这个步骤将输入特征从 in_features 维度映射到 out_features 维度
        support = torch.mm(features, self.weight)

        # 将变换后的特征矩阵与邻接矩阵 adj 相乘；这里使用的是稀疏矩阵乘法 spmm，适用于稀疏邻接矩阵。这一步将节点特征根据邻接关系进行聚合
        output = torch.spmm(adj, support)

        # 最后，如果 active 参数为 True，对结果应用 leaky_relu 激活函数
        if active:
            output = F.leaky_relu(output, negative_slope=0.2)
        return output



class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.2, concat=True):
        """初始化方法"""
        super(GATLayer, self).__init__()
        self.in_channels = in_channels      # 输入特征的维度，即每个节点的特征数量
        self.out_channels = out_channels    # 输出特征的维度，即每个节点的目标特征数量
        self.alpha = alpha                  # LeakyReLU 激活函数的负斜率系数，控制负值的梯度
        self.concat = concat                # 布尔值，决定是否对输出特征应用激活函数并进行拼接

        # self.W：一个可以学习的参数矩阵，大小为 (in_channels, out_channels)，使用 Xavier 均匀分布初始化 W
        self.W = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # self.a：一个可以学习的参数向量，大小为 (2 * out_channels, 1)。这个向量用于计算节点之间的注意力系数。
        self.a = nn.Parameter(torch.zeros(size=(2 * out_channels, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # self.leakyrelu：LeakyReLU 激活函数，用于给注意力系数 𝑒𝑖𝑗 引入非线性。
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """"""
        '''1. 线性变换：对输入特征 ℎ 应用线性变换矩阵 𝑊，得到新的特征矩阵 𝑊ℎ'''
        Wh = torch.mm(h, self.W)  # Shape: (N, out_channels)

        '''2. 计算注意力分数：
           a_input：将每对节点 𝑖 和 𝑗 的特征 𝑊ℎ𝑖 和 𝑊ℎ𝑗 进行拼接，为注意力机制准备输入
           e：通过可学习的参数向量 a 和 LeakyReLU 激活函数，计算每对节点之间的注意力系数 eij。结果 e 的形状为 (N,N)'''
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # Shape: (N, N)

        '''3. 使用邻接矩阵进行注意力掩码：
           zero_vec：一个与 e 形状相同的张量，所有元素设置为一个很小的值（-9e15），用于在 softmax 计算中忽略未连接的节点
           attention：利用邻接矩阵 adj 对注意力系数进行掩码操作，只保留相邻节点之间的注意力系数。
                      非相邻节点的系数被设置为 zero_vec 中的极小值，从而在 softmax 计算时被忽略。
           使用 softmax 对注意力系数进行归一化，使其和为 1。
           '''
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        '''4. 特征聚合：使用计算得到的注意力权重 αij 对节点特征 Wh 进行加权聚合，得到新的节点特征表示 h′'''
        h_prime = torch.matmul(attention, Wh)

        '''5. 输出激活'''
        if self.concat:
            return F.elu(h_prime)  # If concatenation is enabled, apply ELU
        else:
            return h_prime  # Output without activation

    def _prepare_attentional_mechanism_input(self, Wh):
        # Shape (N, out_channels) -> (N, 1, out_channels) -> (N, N, out_channels)
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Shape: (N, N, 2 * out_channels)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_channels)
