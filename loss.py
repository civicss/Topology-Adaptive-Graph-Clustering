# 开发者：李永桢
# 开发时间：2024/9/10 4:22 PM
# 代码功能：对比学习的损失，来自论文 CGC: Contrastive Graph Clustering for Community Detection and Tracking

import torch
import torch.nn.functional as F

def contrastive_loss(h, f_pos, f_neg, W_F, tau=0.5):
    """
    计算基于节点特征的对比损失 L_F
    :param h: 节点的嵌入张量, shape [n, d], 其中 n 是节点数，d 是嵌入维度
    :param f_pos: 正样本节点特征张量, shape [n, d'], 与 h 对应
    :param f_neg: 负样本节点特征张量, shape [n, r, d'], 每个节点有 r 个负样本
    :param W_F: 双线性映射矩阵, shape [d', d]
    :param tau: 温度参数，默认值为 0.5
    :return: 计算的对比损失值
    """
    # 正样本得分
    pos_score = torch.sum(h @ W_F @ f_pos.T, dim=-1) / tau  # h_u^T W_F f'_0_u

    # 负样本得分
    neg_scores = torch.bmm(f_neg, W_F.T @ h.unsqueeze(-1)).squeeze(-1) / tau  # h_u^T W_F f'_v_u for all negatives

    # 拼接正样本和负样本的得分
    all_scores = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)

    # 计算对比损失
    loss = -torch.log_softmax(all_scores, dim=1)[:, 0].mean()  # 只关心正样本的对数概率
    return loss

# 示例使用
n = 5  # 节点数量
d = 16  # 节点嵌入维度
d_prime = 8  # 输入特征维度
r = 3  # 负样本数量
tau = 0.5  # 温度系数

# 模拟一些数据
h = torch.randn(n, d)  # 节点嵌入
f_pos = torch.randn(n, d_prime)  # 正样本特征
f_neg = torch.randn(n, r, d_prime)  # 负样本特征
W_F = torch.randn(d_prime, d)  # 双线性映射矩阵

# 计算对比损失
loss = contrastive_loss(h, f_pos, f_neg, W_F, tau)
print(f'对比损失: {loss}')
