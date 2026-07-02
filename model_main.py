# 开发者：李永桢
# 开发时间：2024/9/23 3:43 PM
# 代码功能：加入了位置编码，自编码器 和 GCN 都加了


import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from GNN import GNNLayer
from torch.nn.parameter import Parameter
import numpy as np
import math

'''1. 自动编码器（AE）模型'''
class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)       # 第一层编码器（全连接层）
        self.enc_2 = Linear(n_enc_1, n_enc_2)       # 第二层编码器
        self.enc_3 = Linear(n_enc_2, n_enc_3)       # 第三层编码器
        self.z_layer = Linear(n_enc_3, n_z)         # 第四层编码器，输出潜在表示

        self.dec_1 = Linear(n_z, n_dec_1)               # 解码器的第一层
        self.dec_2 = Linear(n_dec_1, n_dec_2)           # 解码器的第二层
        self.dec_3 = Linear(n_dec_2, n_dec_3)           # 解码器的第三层
        self.x_bar_layer = Linear(n_dec_3, n_input)     # 解码器的第四层，输出重建的特征

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        # 返回：重建的特征，第一层编码器的输出，第二层编码器的输出，第三层编码器的输出，第四层编码器的输出（即潜在表示）
        return x_bar, enc_h1, enc_h2, enc_h3, z


'''
2. MLP群，关键点：
Linear(n_mlp, 5)：创建了一个线性层，是全连接层
F.leaky_relu：应用 LeakyReLU 激活函数，这是一种在输入小于零时具有小斜率的ReLU激活函数
F.softmax：应用Softmax函数，它将输入的每个元素转换为非负值，并且总和为1，这样输出就可以解释为概率分布
'''
class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 5)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.wl(mlp_in)), dim=1)
        return weight_output

class MLP_1(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_1, self).__init__()
        self.w1 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w1(mlp_in)), dim=1)
        return weight_output

class MLP_2(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_2, self).__init__()
        self.w2 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w2(mlp_in)), dim=1)
        return weight_output

class MLP_3(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_3, self).__init__()
        self.w3 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w3(mlp_in)), dim=1)
        return weight_output


'''3. AGCN 模型'''
# 可学习的位置编码
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super(LearnablePositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim  # 仅存储嵌入维度
        # 初始化可学习的位置编码
        self.position_embeddings = nn.Parameter(torch.randn(1, embedding_dim))

    def forward(self, x):
        n_nodes = x.size(0)  # 获取节点数量
        position_encoding = self.position_embeddings[:n_nodes, :]  # 截取对应长度的位置编码
        alpha = 1
        return x + alpha * position_encoding

# 相对位置编码
class OptimizedRelativePositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super(OptimizedRelativePositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim  # 仅存储嵌入维度

    def generate_relative_encoding(self, n_nodes, adj):
        """
        使用稀疏矩阵避免逐个节点对遍历，优化计算过程
        :param n_nodes: 图的节点数
        :param adj: 稀疏邻接矩阵 (n_nodes, n_nodes)
        :return: 相对位置编码
        """
        adj_matrix = adj.to_dense()  # 转为稠密矩阵
        # 生成节点对的相对位置编码
        row_indices, col_indices = adj_matrix.nonzero(as_tuple=True)  # 获取所有非零元素的位置

        # 通过相对节点位置计算位置编码
        relative_distances = row_indices - col_indices
        position_encoding = torch.zeros(n_nodes, self.embedding_dim, device=adj.device)
        
        # 通过向量化计算位置编码
        for i in range(len(row_indices)):
            distance = relative_distances[i].item()
            # 自定义相对位置编码的方式
            position_encoding[row_indices[i], :] += torch.sin(torch.tensor(distance).float()) * torch.cos(torch.tensor(distance).float())

        return position_encoding

    def forward(self, x, adj):
        # 根据输入张量动态生成相对位置编码
        n_nodes = x.size(0)
        relative_encoding = self.generate_relative_encoding(n_nodes, adj).to(x.device)
        
        return x + relative_encoding  # 返回带有相对位置编码的输出

# 傅立叶位置编码模块
class FourierPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super(FourierPositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim  # 存储嵌入维度

    def generate_position_encoding(self, n_nodes):
        # 生成位置向量
        position = torch.arange(0, n_nodes).unsqueeze(1).float()

        # 傅立叶变换的频率计算
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        
        # 傅立叶编码，正弦和余弦交替
        fourier_encoding = torch.zeros(n_nodes, self.embedding_dim)
        
        # 使用正弦部分进行傅立叶编码
        fourier_encoding[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        # 使用余弦部分进行傅立叶编码
        fourier_encoding[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        
        return fourier_encoding

    def forward(self, x):
        # 根据输入张量动态生成位置编码
        position_encoding = self.generate_position_encoding(x.size(0)).to(x.device)
        
        # 尝试给位置编码加权重: alpha
        alpha = 1
        return x + alpha * position_encoding

# 正弦余弦位置编码模块
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim  # 仅存储嵌入维度

    def generate_position_encoding(self, n_nodes):
        position = torch.arange(0, n_nodes).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        pe = torch.zeros(n_nodes, self.embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # 根据输入张量动态生成位置编码
        position_encoding = self.generate_position_encoding(x.size(0)).to(x.device)
        # 尝试给位置编码加权重: alpha
        alpha = 0.5
        return x + alpha * position_encoding


# 修改后的 AGCN 类
class AGCN(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_input, n_z, n_clusters, ae):
        super(AGCN, self).__init__()
        """自编码器（编码解码各3层），GCN 层（5层），计算注意力的 mlp 层（4个）"""
        '''1. 创建了一个自动编码器实例，该自动编码器用于学习数据的表示'''
        self.ae = ae

        '''2. 创建了 5 个 GCN 层的实例'''
        self.agcn_0 = GNNLayer(n_input, n_enc_1)    # 第一层 GCN，输入是特征矩阵和邻接矩阵
        self.agcn_1 = GNNLayer(n_enc_1, n_enc_2)    # 第二层 GCN，对应 编码器的第一层
        self.agcn_2 = GNNLayer(n_enc_2, n_enc_3)    # 第三层 GCN，对应 编码器的第二层
        self.agcn_3 = GNNLayer(n_enc_3, n_z)        # 第四层 GCN，对应 编码器的第三层，输出层

        '''3. 四个 MLP，用于实现 GCN 层和自编码器之间的注意力：[H_i || Z_i]'''
        self.mlp1 = MLP_1(2 * n_enc_1)    # 第一个 MLP，计算第一层 GCN（agcn_0）和第一层自编码器（n_enc_1）的注意力系数
        self.mlp2 = MLP_2(2 * n_enc_2)    # 第二个 MLP，计算第二层 GCN（agcn_1）和第二层自编码器（n_enc_2）的注意力系数
        self.mlp3 = MLP_3(2 * n_enc_3)    # 第三个 MLP，计算第三层 GCN（agcn_2）和第二层自编码器（n_enc_3）的注意力系数
        self.mlp = MLP_L(3020)            # 第四个 MLP，计算各层 GCN 之间的注意力系数

        '''4. 为每一层创建位置编码模块'''
        # 正弦余弦位置编码
        self.position_encoding_z = SinusoidalPositionalEncoding(10)       # 潜在表示z的位置编码
        self.position_encoding_1 = SinusoidalPositionalEncoding(n_enc_1)  # 第一层位置编码
        self.position_encoding_2 = SinusoidalPositionalEncoding(n_enc_2)  # 第二层位置编码
        self.position_encoding_3 = SinusoidalPositionalEncoding(n_enc_3)  # 第三层位置编码
        self.position_encoding_4 = SinusoidalPositionalEncoding(n_z)      # 第四层位置编码

        # 傅立叶位置编码
        # self.position_encoding_z = FourierPositionalEncoding(10)
        # self.position_encoding_1 = FourierPositionalEncoding(n_enc_1)  # 第一层位置编码
        # self.position_encoding_2 = FourierPositionalEncoding(n_enc_2)  # 第二层位置编码
        # self.position_encoding_3 = FourierPositionalEncoding(n_enc_3)  # 第三层位置编码
        # self.position_encoding_4 = FourierPositionalEncoding(n_z)

        # 可学习的位置编码
        # self.position_encoding_z = LearnablePositionalEncoding(10)
        # self.position_encoding_1 = LearnablePositionalEncoding(n_enc_1)
        # self.position_encoding_2 = LearnablePositionalEncoding(n_enc_2)
        # self.position_encoding_3 = LearnablePositionalEncoding(n_enc_3)
        # self.position_encoding_4 = LearnablePositionalEncoding(n_z)

        # 相对位置编码
        # self.position_encoding_z = OptimizedRelativePositionalEncoding(10)
        # self.position_encoding_1 = OptimizedRelativePositionalEncoding(n_enc_1)  # 第一层位置编码
        # self.position_encoding_2 = OptimizedRelativePositionalEncoding(n_enc_2)  # 第二层位置编码
        # self.position_encoding_3 = OptimizedRelativePositionalEncoding(n_enc_3)  # 第三层位置编码
        # self.position_encoding_4 = OptimizedRelativePositionalEncoding(n_z)


    def forward(self, x, adj):
        """"""
        '''1. 自编码器输出：重建的特征，第一层编码器的输出，第二层编码器的输出，第三层编码器的输出，第四层编码器的输出（即潜在表示）'''
        '''x_bar（n_x, n_input）；h1（n_x, n_enc_1）；h2（n_x, n_enc_2）；h3（n_x, n_enc_3）z（n_x, n_z）'''
        x_bar, h1, h2, h3, z = self.ae(x)

        # 加入位置编码
        h1 = self.position_encoding_1(h1)
        h2 = self.position_encoding_2(h2)
        h3 = self.position_encoding_3(h3)
        z = self.position_encoding_z(z)

        # 加入相对位置编码
        # h1 = self.position_encoding_1(h1, adj)
        # h2 = self.position_encoding_2(h2, adj)
        # h3 = self.position_encoding_3(h3, adj)
        # z = self.position_encoding_z(z, adj)

        x_array = list(np.shape(x))   # 输入数据 x 的形状
        n_x = x_array[0]              # 样本数量

        '''2. AGCN-H：聚合每层自编码器和 GCN 层'''
        '''z1（n_x, n_enc_1）；输入：节点特征，邻接矩阵'''
        z1 = self.agcn_0(x, adj)
        # 加入位置编码
        z1 = self.position_encoding_1(z1)

        '''z2（n_x, n_enc_2）'''
        # 首先，将 z1 和第一个隐藏状态 h1 拼接起来，形成一个二维张量
        # 然后，将这个拼接后的张量输入到第一个 MLP 模块 self.mlp1 中，得到注意力权重 m1（n_x, 2）
        m1 = self.mlp1(torch.cat((h1, z1), 1))
        # 采用 L2 范数归一化，使其每个元素都在单位向量内
        m1 = F.normalize(m1, p=2)
        m11 = torch.reshape(m1[:, 0], [n_x, 1])  # 将 m1 矩阵的第一个列向量 m1[:, 0] 重构成一个形状为 [n_x, 1] 的矩阵 m11
        m12 = torch.reshape(m1[:, 1], [n_x, 1])  # 将 m1 矩阵的第二个列向量 m1[:, 1] 重构成一个形状为 [n_x, 1] 的矩阵 m12
        # 将 m11 和 m12 矩阵分别重复 500 次，使其形状变为 [n_x, 500]
        m11_broadcast = m11.repeat(1, 500)
        m12_broadcast = m12.repeat(1, 500)
        # 使用注意力权重 m11_broadcast 和 m12_broadcast 来分别加权 z1 和 h1，然后将这两个加权后的特征相加
        z2 = self.agcn_1(m11_broadcast.mul(z1) + m12_broadcast.mul(h1), adj)
        # 加入位置编码
        z2 = self.position_encoding_2(z2)

        '''z3'''
        m2 = self.mlp2(torch.cat((h2, z2), 1))      # 拼接，经过 mlp2
        m2 = F.normalize(m2, p=2)                   # L2 归一化
        m21 = torch.reshape(m2[:, 0], [n_x, 1])     # h2 的注意力系数
        m22 = torch.reshape(m2[:, 1], [n_x, 1])     # z2 的注意力系数
        m21_broadcast = m21.repeat(1, 500)          # 扩展列数=500
        m22_broadcast = m22.repeat(1, 500)          # 扩展列数=500
        z3 = self.agcn_2(m21_broadcast.mul(z2) + m22_broadcast.mul(h2), adj)
        # 加入位置编码
        z3 = self.position_encoding_3(z3)

        '''z4'''
        m3 = self.mlp3(torch.cat((h3, z3), 1))      # 拼接，经过 mlp3
        m3 = F.normalize(m3, p=2)                   # L2 归一化
        m31 = torch.reshape(m3[:, 0], [n_x, 1])     # h3 的注意力系数
        m32 = torch.reshape(m3[:, 1], [n_x, 1])     # z3 的注意力系数
        m31_broadcast = m31.repeat(1, 2000)         # 扩展列数=2000
        m32_broadcast = m32.repeat(1, 2000)         # 扩展列数=2000
        z4 = self.agcn_3(m31_broadcast.mul(z3) + m32_broadcast.mul(h3), adj)
        # 加入位置编码
        z4 = self.position_encoding_4(z4)

        '''3. AGCN-S：聚合每个 GCN 层'''
        u = self.mlp(torch.cat((z1, z2, z3, z4, z), 1))     # 拼接，经过 mlp
        u = F.normalize(u, p=2)                             # L2 归一化
        u0 = torch.reshape(u[:, 0], [n_x, 1])               # z1 的注意力系数
        u1 = torch.reshape(u[:, 1], [n_x, 1])               # z2 的注意力系数
        u2 = torch.reshape(u[:, 2], [n_x, 1])               # z3 的注意力系数
        u3 = torch.reshape(u[:, 3], [n_x, 1])               # z4 的注意力系数
        u4 = torch.reshape(u[:, 4], [n_x, 1])               # z 的注意力系数

        tile_u0 = u0.repeat(1, 500)         # 扩展列数=500
        tile_u1 = u1.repeat(1, 500)         # 扩展列数=500
        tile_u2 = u2.repeat(1, 2000)        # 扩展列数=2000
        tile_u3 = u3.repeat(1, 10)          # 扩展列数=10
        tile_u4 = u4.repeat(1, 10)          # 扩展列数=10

        # 拼接，经过注意力系数
        net_output = torch.cat((tile_u0.mul(z1), tile_u1.mul(z2), tile_u2.mul(z3), tile_u3.mul(z4), tile_u4.mul(z)), 1)
        '''
        # 在最后一个图卷积网络层中，将拼接后的 net_output 并传递给该层。这里 active=False 指示不应用激活函数
        net_output = self.agcn_z(net_output, adj, active=False)
        # 输出 n_x * n_clusters 的节点特征矩阵，通过 softmax 函数，得到节点属于每个类的概率矩阵
        predict = F.softmax(net_output, dim=1)
        '''

        '''
        5.返回值：
          x_bar（n_x, n_input）：重建的特征；  q（n_x, n_clusters）：聚类分布结果，根据自编码器的编码器输出节点潜在表示 z 和 聚类中心的距离来分类
          predict（n_x, n_clusters）：GCN 的预测结果；    z（n_x, n_z）：第四层编码器的输出，即潜在表示
          net_output（n_x, 3020）：拼接后的结果
        '''
        return x_bar, net_output


'''4. AGCN_MA 模型：用于加权注意力融合三种 motif 对应的邻接矩阵'''
'''3种 motif 计算注意力系数的 mlp'''
class MLP_MA(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_MA, self).__init__()
        self.w = Linear(n_mlp, 3)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w(mlp_in)), dim=1)
        return weight_output

'''ZQ 计算注意力系数的 mlp'''
class MLP_ZQ(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_ZQ, self).__init__()
        self.w = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w(mlp_in)), dim=1)
        return weight_output

'''加权的多元交叉熵损失: Binary Cross-Entropy Loss'''
def BCE(out, tar, weight):
    # 输入: out: 模型的预测输出;  tar: 目标标签;  weight: 用于加权损失的张量，通常用于处理不平衡的数据集;
    # eps: 设置一个小的常量（1e-12），防止在计算对数时出现nan情况
    eps = 1e-12
    l_n = weight * (tar * (torch.log(out+eps)) + (1 - tar) * (torch.log(1 - out+eps)))
    l = -torch.sum(l_n) / torch.numel(out)
    return l

'''生成目标分布 P'''
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

'''AGCN_MA 模型'''
class AGCN_MA(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v, pretrain_path='pkl'):
        super(AGCN_MA, self).__init__()
        """"""
        '''1. 创建了一个自动编码器实例，该自动编码器用于学习数据的表示'''
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        '''2. 使用预训练的模型参数（从 args.pretrain_path 路径加载）来替换自编码器的初始参数'''
        self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'))

        '''
          3.簇的中心：这是一个参数化的层，用于存储每个簇的中心。这些中心在训练过程中会通过梯度下降进行优化。
            self.cluster_layer 是一个形状为 [n_clusters, n_z] 的矩阵，其中每个行向量代表一个簇的中心。
            nn.Parameter 将张量转换为模型参数，即这个张量会被自动添加到模型的参数列表中，
            并且在调用 model.parameters() 时可以被优化器所识别和更新。
        '''
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))  # 聚类中心向量矩阵
        torch.nn.init.xavier_normal_(self.cluster_layer.data)          # 初始化矩阵

        '''4. 三个 AGCN 层'''
        self.agcn1 = AGCN(
            500, 500, 2000,                     # 自编码器的 编码器 和 解码器 每层的输出特征维度
            n_input=n_input,                    # 输入特征维度
            n_z=n_z,                            # 自编码器的 编码器 输出特征维度
            n_clusters=n_clusters,              # 聚类中心数量
            ae=self.ae,                         # 自编码器
        )
        self.agcn2 = AGCN(
            500, 500, 2000,                     # 自编码器的 编码器 和 解码器 每层的输出特征维度
            n_input=n_input,                    # 输入特征维度
            n_z=n_z,                            # 自编码器的 编码器 输出特征维度
            n_clusters=n_clusters,              # 聚类中心数量
            ae=self.ae,                         # 自编码器
        )
        self.agcn3 = AGCN(
            500, 500, 2000,                     # 自编码器的 编码器 和 解码器 每层的输出特征维度
            n_input=n_input,                    # 输入特征维度
            n_z=n_z,                            # 自编码器的 编码器 输出特征维度
            n_clusters=n_clusters,              # 聚类中心数量
            ae=self.ae,                         # 自编码器
        )

        '''5. 聚合每个 AGCN'''
        self.mlp = MLP_MA(3 * 3020)             # MLP，计算各层 AGCN 之间的注意力系数

        '''6. 聚合 QZ'''
        self.mlp_ZQ = MLP_ZQ(2 * n_clusters)

        '''7. 一个 GCN 层，输入带注意力系数的 AGCN 输出，输出聚类的概率'''
        self.agcn_motif = GNNLayer(3 * 3020, n_clusters)

        '''8. 一个正数超参数，用于调节分配概率的平滑程度'''
        self.v = v
        self.n_clusters = n_clusters

    def forward(self, data, adj1, adj2, adj3, adj):
        """
        data: 特征
        adj1 ~ adj3: 三种 motif 对应的邻接矩阵
        adj: 带有三个 motif 的邻接矩阵
        """
        '''1. 自编码器输出：重建的特征，第一层编码器的输出，第二层编码器的输出，第三层编码器的输出，第四层编码器的输出（即潜在表示）'''
        '''z（n_x, n_z）'''
        x_array = list(np.shape(data))  # 输入数据 x 的形状
        n_x = x_array[0]                # 样本数量
        _, _, _, _, z = self.ae(data)

        '''2. 聚类分布结果 q 的计算：根据自编码器的节点潜在表示 z 和聚类中心的聚类来分类'''
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        '''3. 各层 AGCN 输出结果：
           x_bar（n_x, n_input）：重建的特征；
           net_output（n_x, n_clusters）：未经过归一化处理的 GCN 预测结果；'''
        x_bar1, net_output1 = self.agcn1(data, adj1)
        x_bar2, net_output2 = self.agcn2(data, adj2)
        x_bar3, net_output3 = self.agcn3(data, adj3)

        # 对每层的输出应用 dropout
        # net_output1 = self.dropout(net_output1)
        # net_output2 = self.dropout(net_output2)
        # net_output3 = self.dropout(net_output3)

        '''4. 拼接注意力系数'''
        u_combined = self.mlp(torch.cat((net_output1, net_output2, net_output3), 1))  # 拼接经过 MLP
        u_combined = F.normalize(u_combined, p=2)                                     # L2 归一化

        '''5. 获取每个输出的注意力系数'''
        u1 = torch.reshape(u_combined[:, 0], [n_x, 1])  # net_output1 的注意力系数
        u2 = torch.reshape(u_combined[:, 1], [n_x, 1])  # net_output2 的注意力系数
        u3 = torch.reshape(u_combined[:, 2], [n_x, 1])  # net_output3 的注意力系数

        '''6. 扩展列数'''
        tile_u1 = u1.repeat(1, 3020)
        tile_u2 = u2.repeat(1, 3020)
        tile_u3 = u3.repeat(1, 3020)

        '''7. 根据注意力系数加权合并'''
        net_output_combined = torch.cat((tile_u1.mul(net_output1), tile_u2.mul(net_output2), tile_u3.mul(net_output3)), 1)

        '''8. 将合并后的输出传递给 GCN，net_output：n_x * n_clusters'''
        net_output = self.agcn_motif(net_output_combined, adj, active=False)

        '''9. 输出 n_x * n_clusters 的节点特征矩阵，使用 softmax 函数得到概率矩阵'''
        predict = F.softmax(net_output, dim=1)

        '''10. Q（自编码器）Z（GCN） 的融合结果：'''
        p_ZH = self.mlp_ZQ(torch.cat((predict, q), 1))               # 拼接经过 MLP
        p_ZH = F.normalize(p_ZH, p=2)                                # L2 归一化
        p_ZH1 = torch.reshape(p_ZH[:, 0], [n_x, 1])                  # GCN 的注意力系数
        p_ZH2 = torch.reshape(p_ZH[:, 1], [n_x, 1])                  # 自编码器的注意力系数
        p_ZH1_broadcast = p_ZH1.repeat(1, self.n_clusters)           # 扩展列数
        p_ZH2_broadcast = p_ZH2.repeat(1, self.n_clusters)           # 扩展列数
        z_F = p_ZH1_broadcast.mul(predict) + p_ZH2_broadcast.mul(q)  # 计算predict和q的加权和，得到一个新的表示z_F
        # 应用softmax函数在第一个维度（通常是类别维度）上对z_F进行归一化，使得z_F的每一行加起来等于1，这样每一行都可以解释为概率分布
        z_F = F.softmax(z_F, dim=1)

        '''11. 硬自监督损失函数'''
        '''
        例: 1)归一化后的z_F=[[0.1, 0.7, 0.2],   2)得到的聚类分配[1,0,0,0,2]   3)独热编码形式clu_assignment_onehot=[[0 1 0],
                           [0.4, 0.4, 0.2],                                                                  [1 0 0],
                           [0.9, 0.05, 0.05],                                                                [1 0 0],
                           [0.6, 0.1, 0.3],                                                                  [1 0 0],
                           [0.01, 0.01, 0.98]]                                                               [0 1 1]]
            4)阈值矩阵thres_matrix=[[0.8, 0.8, 0.8], ..., [0.8, 0.8, 0.8]]
            5)权重标签矩阵 weight_label=[[0, 0, 0],
                                       [0, 0, 0],
                                       [1, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 1]]
            6)多元交叉熵的计算：对于第三个样本，真实标签 y3=[1, 0, 0], 预测概率p3=[0.9, 0.05, 0.05],
                             损失计算为: L3 = -(1⋅log(0.9)+0⋅log(0.05)+0⋅log(0.05)) = -log(0.9)
        '''
        # 1.聚类分配：取概率分布 z_F 的最大值索引来为每个样本确定聚类分配，-1是最后一个维度
        clu_assignment = torch.argmax(z_F, -1)
        # 2.独热编码矩阵形式：将聚类分配转换为独热编码形式，使每个样本在其所属聚类的索引位置为1，其他位置为0
        clu_assignment_onehot = F.one_hot(clu_assignment, self.n_clusters)
        thres = 0.8                                                 # 设置高置信度样本阈值0.8
        thres_matrix = torch.zeros_like(z_F) + thres                # 阈值矩阵，和 z_F 形状相同
        # 3.计算权重标签：torch.ge 是一个比较函数，它会返回一个布尔矩阵，表示 normalize(z_F) 中的每个元素是否大于或等于 thres_matrix 中的对应元素
        #               例如，如果某个样本的归一化概率值大于等于0.8，则在该位置的布尔值为 True（或1），否则为 False（或0）
        weight_label = torch.ge(F.normalize(z_F, p=2), thres_matrix).type(torch.FloatTensor).to(z_F.device)
        # 4.计算伪标签损失：weight_label 作为权重，确保只有那些高置信度的聚类分配（即那些概率值大于或等于阈值的集群）会贡献到损失中
        pseudo_label_loss = BCE(z_F, clu_assignment_onehot, weight_label)

        '''12. 返回值：
               x_bar1-3: 每层 AGCN 的 AE 重建的节点特征矩阵
               q: 自编码器得到的聚类分布结果
               z: 自编码器的节点潜在表示
               predict: GCN 得到的聚类分布结果(经过3种 motif 聚合)
               z_F: 自编码器 和 GCN 融合的聚类分布结果
               pseudo_label_loss: z_F 作为伪标签的损失值
        '''
        return x_bar1, x_bar2, x_bar3, q, z, predict, z_F, pseudo_label_loss