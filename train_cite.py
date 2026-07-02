# 开发者：李永桢
# 开发时间：2024/9/23 5:21 PM
# 代码功能：原始图 + 1/3 motif; P 基于 Z 增强 1 次, Q 和 QZ 增强 1 次; lr = 1e-4; epoch = 150; 参数见 initial();



import os
import argparse
import random
from torch.optim import Adam
from utils import load_data, load_graph, Graph_Construction
from draw import plot_similarity_matrix
from GNN import GNNLayer
from evaluation import eva, evao, cluster_acc
from model_4_1 import *
from sklearn.cluster import KMeans
import time

'''1. 初始化'''
def initial():
    if args.name == 'cite':
        args.file_name = './output_4/output_4_cite/cite_1.txt'  # 结果文件名字
        args.lr = 1e-4                      # 学习率
        args.k = None                       # 是否构建 KNN 图
        args.n_clusters = 6                 # 聚类类别数目
        args.n_input = 3703                 # 输入特征维度
        args.enhance = False                # 是否根据图高阶相似性进行图增强
        args.epoch = 160                    # 训练轮次

        args.ld1 = 0.1                # 损失1: QP KL散度的权重
        args.ld2 = 0.1                # 损失1: ZP KL散度的权重
        args.ld3 = 0.01               # 损失1: QZ KL散度的权重
        args.ld4 = 1                  # 损失2: 特征重构损失的权重
        args.ld5 = 0.001              # 损失2: 结构重构损失的权重
        args.ld6 = 0.1                # 损失3: 相似性损失的权重
        args.ld7 = 0.0005             # 损失4: 提示损失1: 节点与聚类中心均方差
        args.ld8 = 0.0005             # 损失4: 提示损失2: 节点与聚类中心交叉熵
        args.ld9 = 0                  # 损失5: 聚类中心正交性约束损失的权重
        args.ld10 = 1                 # 损失6: 高置信度样本伪标签自监督损失的权重

        # args.ld1 = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        # args.ld2 = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        # args.ld3 = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        # args.ld4 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # args.ld5 = nn.Parameter(torch.tensor(0.001), requires_grad=True)
        # args.ld6 = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        # args.ld7 = nn.Parameter(torch.tensor(5e-4), requires_grad=True)
        # args.ld8 = nn.Parameter(torch.tensor(5e-4), requires_grad=True)
        # args.ld10 = nn.Parameter(torch.tensor(1.0), requires_grad=True)


'''3. 生成目标分布 P'''
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T  # 转置新版本的写法

'''4. 结构重构损失: size_average=False, 这意味着损失值不会自动求平均，而是对所有样本的损失值进行求和。因此需要手动归一化'''
bce_loss = torch.nn.BCELoss(size_average=False)

'''5. 相似化损失: 通过计算每个节点与其邻居节点在两个嵌入空间中的欧式距离平方，并将这些距离累加起来得到的。
      最终，损失值被归一化，以便于后续的优化过程。'''
def loss_Similarity(C, Q, Z, QZ, device):
    """
    计算两个嵌入空间中节点对的相似性损失。

    :参数 C: 每个节点的邻居列表，长度为 len1, 这些节点集合定义了哪些节点应该彼此保持相似。
            [[8,12],[2,4],[5,7]]
    :参数 Z: 节点嵌入矩阵(N * m), 其中 N 是节点数, m 是每个节点的嵌入维度。表示第一种嵌入空间。
    :参数 Z1: 另一个节点嵌入矩阵，表示第二种嵌入空间，维度与 Z 相同。
    :返回值 loss: 损失值 loss
    :返回值 edge_num: 边数
    """
    # 将嵌入矩阵移动到指定设备
    Q = Q.to(device)
    Z = Z.to(device)
    QZ = QZ.to(device)

    # 初始化损失和边数
    loss = torch.zeros(1, device=device)
    edge_num = 0

    # 计算损失
    for i, neighbors in enumerate(C):
        edge_num += len(neighbors)
        for j in neighbors:
            loss += torch.sum((Q[i] - Q[j])**2)
            loss += torch.sum((Z[i] - Z[j])**2)
            loss += torch.sum((QZ[i] - QZ[j])**2)

    return loss, edge_num



def reg_loss(C, Z, Z1, device):
    """
    :参数 C: 每个节点的邻居列表，长度为 len1，这些节点集合定义了哪些节点应该彼此保持相似。
            [[8,12],[2,4],[5,7]]
    :参数 Z: 节点嵌入矩阵（N × m），其中 N 是节点数，m 是每个节点的嵌入维度。表示第一种嵌入空间。
    :参数 Z1: 另一个节点嵌入矩阵，表示第二种嵌入空间，维度与 Z 相同。
    :返回值 loss: 损失值 loss
    :返回值 edge_num: 边数
    """
    '''1. 初始化'''
    # 获取 Z 的维度，n 是节点数量，m 是嵌入维度
    n, m = Z.shape

    # 初始化 loss 为一个标量张量，初始化为 0，并移动到合适的设备（CPU 或 GPU）
    len1 = len(C)
    loss = torch.zeros(1)
    loss = torch.FloatTensor(loss)
    loss = loss.to(device)

    # edge_num 用于计算集合中节点对的总数
    edge_num = 0.0

    '''2. 计算损失'''
    for i in range(0, len1):
        # 遍历 C 中的每个子集合（C[i]），len2 是当前子集合的长度
        len2 = len(C[i])

        # edge_num 用于记录所有节点对的总数量，用于后续归一化
        edge_num = edge_num + len2

        # 对每个子集合，计算每个节点 i 与该子集合中的其他节点 C[i][j] 之间的欧式距离平方（torch.square），
        # 并将这些距离在两个嵌入空间（Z 和 Z1）中进行累加。
        for j in range(0, len2):
            loss += torch.sum(torch.square(Z[i] - Z[C[i][j]]))
            loss += torch.sum(torch.square(Z1[i] - Z1[C[i][j]]))

    '''3. 返回损失值和节点对的总数量（edge_num）'''
    return loss, edge_num



'''6. 训练函数'''
def train_AGCN(dataset):
    """"""

    '''1. AGCN_MA 模型'''
    model = AGCN_MA(
        500, 500, 2000, 2000, 500, 500,     # 自编码器的 编码器 和 解码器 每层的输出特征维度
        n_input=args.n_input,               # 输入特征维度
        n_z=args.n_z,                       # 自编码器的 编码器 输出特征维度
        n_clusters=args.n_clusters,         # 聚类中心数量
        v=args.v,                           # 一个正数超参数，用于调节分配概率的平滑程度
        pretrain_path=args.pretrain_path    # 自编码器预训练路径
    ).to(device)

    '''2. 优化器: 优化算法Adam, 参数包括模型参数、学习率、weight_decay 参数来实现 L2 正则化。它会在每次更新权重时对其施加惩罚。'''
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4)

    '''3. 加载图数据C: [[8,12],[2,4],[5,7]]，新矩阵，新图的节点对应的邻居集合，原始矩阵；然后将张量移动到指定设备'''
    adj_3, _, C_3, _, adjp = load_graph(args.name, k=args.k, enhance=args.enhance, motif_num=3)
    adj_3 = adj_3.to(device)
    adjp = adjp.to(device)

    adj_4, _, C_4, _, _ = load_graph(args.name, k=args.k, enhance=args.enhance, motif_num=4)
    adj_4 = adj_4.to(device)

    adj_5, _, C_5, _, _ = load_graph(args.name, k=args.k, enhance=args.enhance, motif_num=5)
    adj_5 = adj_5.to(device)

    adj, _, C, _, _ = load_graph(args.name, k=args.k, enhance=args.enhance, motif_num=None)
    adj = adj.to(device)

    '''4. 自编码器辅助的 K均值聚类 初始化'''
    # 1.获得节点潜在表示矩阵：向自编码器输入“原始特征矩阵”，输出“节点潜在表示”，用这个表示进行 K-means 聚类
    data = torch.Tensor(dataset.x).to(device)  # 特征
    y = dataset.y                              # 标签
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)         # 自编码器的编码器输出的节点的潜在表示

    # 2.初始化 K 均值聚类模型：n_clusters 聚类中心数目；n_init：初始化的次数，较高的 n_init 值可以增加找到更好聚类中心的机会
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)

    # 3.预测聚类标签
    # kmeans.fit_predict()：这是 scikit-learn 的 KMeans 类的一个方法，它结合了 fit 和 predict 方法。具体来说：
    # fit 方法用于在数据集上训练模型，找到质心。predict 方法用于预测每个样本的簇标签
    # fit_predict 方法会先执行 fit，然后直接使用训练好的模型对数据进行预测，并将预测结果（即每个样本的簇标签）返回
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())

    # 4.更新模型中的聚类层：更新为 kmeans 的聚类中心，kmeans.cluster_centers_（簇的数量，原始数据的特征数量）
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    # 5.评估聚类结果
    best_acc_QZ1, best_nmi_QZ1, best_ari_QZ1, best_f1_QZ1 = eva(y, y_pred, 'pae')
    loss_weights = {}           # 创建一个字典来存储损失项权重
    for i in range(1, 11):      # 初始化字典中的所有键为 0
        loss_weights[f'ld{i}'] = 0



    '''5. 模型训练'''
    for epoch in range(args.epoch):

        '''1. 模型输出：
                x_bar1-3: 每层 AGCN 的 AE 重建的节点特征矩阵
                q: 自编码器得到的聚类分布结果
                h: 自编码器的节点潜在表示
                predict: GCN 得到的聚类分布结果(经过3种 motif 聚合)
                QZ: 自编码器 和 GCN 融合的聚类分布结果
                pl_loss: z_F 作为伪标签的损失值
        '''
        x_bar1, x_bar2, x_bar3, q, h, predict, QZ, pl_loss = model(data, adj_3, adj_4, adj_5, adj)
        tmp_q = q.detach()  # 每个簇的概率（n * n_cluster_num），新的版本中建议使用 detach() 获取张量的原始数据
        
        '''生成目标分布 P, 分布 P 的预测结果'''
        # 计算新的目标分布 P
        if epoch <= 50:
            p = target_distribution(tmp_q)    
        elif epoch <= 75:
            p = target_distribution(predict.data)
        else:
            p = target_distribution(tmp_q)
        
        res_P = p.data.cpu().numpy().argmax(1)         # P：目标分布 P 的预测结果

        '''其他分布的预测结果'''
        res_Q = tmp_q.cpu().numpy().argmax(1)          # Q：编码器的预测结果
        res_Z = predict.data.cpu().numpy().argmax(1)   # Z：GCN 的预测结果
        res_QZ = QZ.data.cpu().numpy().argmax(1)       # QZ：自编码器 和 GCN 的预测结果的融合

        '''考虑增强 Z: 1次'''
        predict_enhance = target_distribution(predict.detach())
        res_Z_enhance = predict_enhance.data.cpu().numpy().argmax(1)

        '''考虑增强 QZ: 1次'''
        QZ_enhance = target_distribution(QZ.detach())
        res_QZ_enhance = QZ_enhance.data.cpu().numpy().argmax(1)


        '''2. 评估预测结果：自编码器编码器的预测结果'''
        if epoch % 1 == 0:
            '''输出到文件1'''
            # 假设 evao() 函数返回一个字符串或者数字结果
            acc_Q, nmi_Q, ari_Q, f1_Q = evao(y, res_Q, str(epoch) + 'Q')
            acc_P, nmi_P, ari_P, f1_P = evao(y, res_P, str(epoch) + 'P')
            acc_Z, nmi_Z, ari_Z, f1_Z = evao(y, res_Z, str(epoch) + 'Z')
            acc_Z1, nmi_Z1, ari_Z1, f1_Z1 = evao(y, res_Z_enhance, str(epoch) + 'Z增强')
            acc_QZ, nmi_QZ, ari_QZ, f1_QZ = evao(y, res_QZ, str(epoch) + 'QZ')
            acc_QZ1, nmi_QZ1, ari_QZ1, f1_QZ1 = evao(y, res_QZ_enhance, str(epoch) + 'QZ增强')
            print()

            # 打开文件并将结果写入，'a' 模式表示追加内容到文件末尾
            with open(args.file_name, 'a') as file:
                file.write(f"Epoch {epoch} Results:\n")
                file.write(f"Q Result: acc={acc_Q:.4f}, nmi={nmi_Q:.4f}, ari={ari_Q:.4f}, f1={f1_Q:.4f}\n")
                file.write(f"P Result: acc={acc_P:.4f}, nmi={nmi_P:.4f}, ari={ari_P:.4f}, f1={f1_P:.4f}\n")
                file.write(f"Z Result: acc={acc_Z:.4f}, nmi={nmi_Z:.4f}, ari={ari_Z:.4f}, f1={f1_Z:.4f}\n")
                file.write(f"Z_enhance Result: acc={acc_Z1:.4f}, nmi={nmi_Z1:.4f}, ari={ari_Z1:.4f}, f1={f1_Z1:.4f}\n")

                file.write(f"QZ Result: acc={acc_QZ:.4f}, nmi={nmi_QZ:.4f}, ari={ari_QZ:.4f}, f1={f1_QZ:.4f}\n")
                file.write(f"QZ_enhance Result: acc={acc_QZ1:.4f}, nmi={nmi_QZ1:.4f}, ari={ari_QZ1:.4f}, f1={f1_QZ1:.4f}\n")
                file.write("\n")  # 添加换行符以分隔不同的epoch结果

            # 最佳结果
            if acc_QZ1 > best_acc_QZ1:
                best_acc_QZ = acc_QZ
                best_nmi_QZ = nmi_QZ
                best_ari_QZ = ari_QZ
                best_f1_QZ = f1_QZ

                best_acc_QZ1 = acc_QZ1
                best_nmi_QZ1 = nmi_QZ1
                best_ari_QZ1 = ari_QZ1
                best_f1_QZ1 = f1_QZ1

                # 更新参数值
                for i in range(10):
                    loss_weights[f'ld{i+1}'] = getattr(args, f'ld{i+1}')

        '''3. 在每个 epoch 或某个特定操作后释放 GPU 内存：
                      检查 torch.cuda 是否具有 empty_cache 方法，如果有，则调用它清空 CUDA 内存缓存'''
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        '''4. 损失'''
        '''损失1: KL 散度'''
        # 自编码器输出 Q 和 目标分布 P 的KL散度：q.log() 是 q 的对数，p 是目标分布，这里使用批平均（batchmean）来计算
        loss_kl_QP = F.kl_div(q.log(), p, reduction='batchmean')

        # GCN输出 Z 和 目标分布 P 的 KL散度：pred.log() 是 pred 的对数
        loss_kl_ZP = F.kl_div(predict.log(), p, reduction='batchmean')

        # 自编码器输出 Q 和 GCN 输出 Z 的 KL散度
        loss_kl_QZ = F.kl_div(q.log(), predict, reduction='batchmean')

        '''损失2: 重构损失'''
        # 特征重构损失：均方误差，重建输出和原始特征
        loss_fres_1 = F.mse_loss(x_bar1, data)
        loss_fres_2 = F.mse_loss(x_bar2, data)
        loss_fres_3 = F.mse_loss(x_bar3, data)

        # 结构重构损失：节点嵌入内积通过sigmoid()作为重构邻接矩阵，与原始邻接矩阵计算交叉熵损失
        graph_re = Graph_Construction(QZ)       # 根据图表示重构的邻接矩阵
        adj_re = graph_re.Middle().to(device)   # 调用图构建对象中的 Middle 方法，可能用于获取重构的图的邻接矩阵或特征表示。
        N, _ = adj_re.shape                     # 节点数量 N
        loss_sres = 1 / (N * N) * bce_loss(adj_re, adjp)   # 结构重构损失

        '''损失3: 相似性损失：一个节点和其邻居节点应该具有相似的概率分布，计算两个节点的欧式距离，然后求和归一化'''
        '''可以考虑选取三角形、四边形、五边形 motif 生成的图中的一种图来计算相似度损失'''
        # 基于三角形、四边形、五边形 motif 生成的图的相似性损失
        # loss_sim_3, edge_num_3 = loss_Similarity(C_3, q, predict, QZ, device=args.device)
        # loss_sim_3 = loss_sim_3 / edge_num_3
        # loss_sim_3 = loss_sim_3.item()

        # loss_sim_4, edge_num_4 = loss_Similarity(C_4, q, predict, QZ, device=args.device)
        # loss_sim_4 = loss_sim_4 / edge_num_4
        # loss_sim_4 = loss_sim_4.item()

        # loss_sim_5, edge_num_5 = loss_Similarity(C_5, q, predict, QZ, device=args.device)
        # loss_sim_5 = loss_sim_5 / edge_num_5
        # loss_sim_5 = loss_sim_5.item()

        loss_sim_3, edge_num3 = reg_loss(C_3, q, predict, device=args.device)
        loss_sim_3 = loss_sim_3 / edge_num3
        loss_sim_3 = loss_sim_3.squeeze()

        loss_sim_4, edge_num4 = reg_loss(C_4, q, predict, device=args.device)
        loss_sim_3 = loss_sim_3 / edge_num4
        loss_sim_4 = loss_sim_3.squeeze()

        loss_sim_5, edge_num5 = reg_loss(C_5, q, predict, device=args.device)
        loss_sim_5 = loss_sim_5 / edge_num5
        loss_sim_5 = loss_sim_5.squeeze()


        '''损失4: 提示损失: 1.均方误差损失; 2.交叉熵损失;'''
        '''可以考虑这里的提示损失的节点嵌入替换为 GCN 的输出'''
        # prompt 是从模型的聚类层提取的聚类中心，形状为(n_clusters, n_z) (3, 10)
        # h 是自编码器的输出(节点的潜在表示), 形状为(n, n_z) (3025, 10)
        # res_Q 是自编码器的预测结果，形状为(n, ) (3025, )
        # prompt[res_Q] 是节点的预测类别的中心的嵌入，形状为(n, n_z) (3025, 10)
        prompt = model.cluster_layer

        # 1.计算节点嵌入 h 和节点对应聚类中心 prompt[res1] 之间的均方误差损失，即认为节点嵌入和预测聚类中心嵌入应该一致
        loss_prompt_1 = F.mse_loss(h, prompt[res_Q])

        # 2.计算交叉熵损失: 促进节点嵌入和预测聚类中心嵌入更一致
        # 交叉熵损失第一项torch.mm(h, prompt.T): 计算嵌入 h 与聚类中心的转置之间的矩阵乘法，得到一个形状为 (3025, n_clusters) 的相似度矩阵，
        # 表示每个嵌入与所有聚类中心的相似度。这个相似度矩阵在交叉熵损失计算时实际上会经过 softmax 转换为概率分布。
        # 交叉熵损失第二项: 每个节点的预测标签。
        # 交叉熵损失计算过程: 1.Softmax: 对相似度矩阵的每一行应用 softmax，得到每个样本对应各聚类中心的预测概率分布。
        #                  2.计算损失: 对于每个样本，交叉熵损失计算模型预测的概率分布与真实标签之间的差距。
        # 具体来说，对于样本 i, 计算的是：lossi=−log(ptrue), 其中 ptrue 是样本 i 的正确聚类中心的预测概率。
        loss_prompt_2 = F.cross_entropy(
                torch.mm(h, prompt.T), torch.LongTensor(res_Q).view(-1, ).to(device))
        
        '''损失5: 聚类中心的正交性约束'''
        # torch.mm(prompt, prompt.T): 聚类中心的内积，即聚类中心的相似度
        # torch.eye(prompt.shape[0]: 对角线全1的单位矩阵
        # torch.norm(...): 计算输入矩阵的 Frobenius 范数，即矩阵中所有元素的平方和再开根号。它用于衡量矩阵与单位矩阵之间的距离。
        # 这个约束损失的意义在于：它测量聚类中心之间的内积矩阵与单位矩阵之间的差距。如果聚类中心是正交的（即彼此之间没有相似性），那么内积矩阵应该接近于单位矩阵。
        loss_orthogonality = torch.norm(torch.mm(prompt, prompt.T) - torch.eye(prompt.shape[0]).to(device))

        
        '''总的损失'''
        loss = args.ld1 * loss_kl_QP + args.ld2 * loss_kl_ZP + args.ld3 * loss_kl_QZ + \
               args.ld4 * loss_fres_1 + args.ld5 * loss_sres + \
               args.ld6 * (loss_sim_3 + loss_sim_4 + loss_sim_5) + \
               args.ld7 * loss_prompt_1 + args.ld8 * loss_prompt_2 + \
               args.ld9 * loss_orthogonality + \
               args.ld10 * pl_loss


        '''6. 反向传播'''
        print(f'epoch={epoch}')
        # print(f'参数: args.ld1={args.ld1.item()}, args.ld2={args.ld2.item()}, args.ld3={args.ld3.item()}, '
        #       f'args.ld4={args.ld4.item()}, args.ld5={args.ld5.item()}, args.ld6={args.ld6.item()}, '
        #       f'args.ld7={args.ld7.item()}, args.ld8={args.ld8.item()}, args.ld9={args.ld9}, '
        #       f'args.ld10={args.ld10.item()}')
        print(f'KL散度: QP_KL散度={loss_kl_QP}, ZP_KL散度={loss_kl_ZP}, QZ_KL散度={loss_kl_QZ};')
        print(f'重构损失: 特征重构损失={loss_fres_1}、{loss_fres_2}、{loss_fres_3}; 结构重构损失={loss_sres};')
        print(f'相似性损失={loss_sim_3}、{loss_sim_4}、{loss_sim_5};')
        print(f'提示损失1(节点与聚类中心均方差)={loss_prompt_1}, 提示损失2(节点与聚类中心交叉熵)={loss_prompt_2};')
        print(f'聚类中心正交性损失={loss_orthogonality}; 高置信度样本的伪标签自监督损={pl_loss}')
        optimizer.zero_grad()       # 梯度清零
        loss.backward()             # 计算损失函数（loss）关于模型参数的梯度
        optimizer.step()            # 参数更新

    '''6. 记录最佳结果，'a' 模式表示追加内容到文件末尾'''
    with open(args.file_name, 'a') as file:
        file.write(f"Best Results:\n")
        # 遍历字典中的每个键值对
        for key, value in loss_weights.items():
            # 写入中文说明和对应的值
            file.write(f'参数{key}的权重是: {value}\n\n')
        file.write(f"QZ Result: acc={best_acc_QZ:.4f}, nmi={best_nmi_QZ:.4f}, ari={best_ari_QZ:.4f}, f1={best_f1_QZ:.4f}\n")
        file.write(f"QZ增强 Result: acc={best_acc_QZ1:.4f}, nmi={best_nmi_QZ1:.4f}, ari={best_ari_QZ1:.4f}, f1={best_f1_QZ1:.4f}\n")
        file.write("\n")



'''主函数'''
if __name__ == "__main__":

    '''1. 创建命令行解析器'''
    parser = argparse.ArgumentParser(
        # 描述，当命令行工具使用--h或--help选项时，这个描述会显示在帮助信息的开始部分，告诉用户这个脚本的用途
        description='train',
        # 一个帮助消息格式化类，它会为每个参数显示其默认值
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 添加超参数：根据每个数据集不同需要改的
    parser.add_argument('--name', type=str, default='cite')                 # 数据集名字
    parser.add_argument('--file_name', type=str, default='cite')            # 输出结果文件名字
    parser.add_argument('--enhance', type=bool, default=False)              # 是否进选择经过相似性增强的图
    parser.add_argument('--module_name', type=str, default='m4')            # 模型名字
    parser.add_argument('--k', type=int, default=None)                      # k 近邻的取值
    parser.add_argument('--n_clusters', default=4, type=int)                # 聚类的数目
    parser.add_argument('--n_z', default=10, type=int)                      # 自编码器的 编码器 输出特征维度
    # parser.add_argument('--ld1', type=float, default=0.1)            # 损失1: QP KL散度的权重
    # parser.add_argument('--ld2', type=float, default=0.1)            # 损失1: ZP KL散度的权重
    # parser.add_argument('--ld3', type=float, default=0.1)            # 损失1: QZ KL散度的权重
    # parser.add_argument('--ld4', type=float, default=0.1)            # 损失2: 特征重构损失的权重
    # parser.add_argument('--ld5', type=float, default=0.1)            # 损失2: 结构重构损失的权重
    # parser.add_argument('--ld6', type=float, default=0.1)            # 损失3: 相似性损失的权重
    # parser.add_argument('--ld7', type=float, default=0.1)            # 损失4: 提示损失的权重: 节点与聚类中心均方差
    # parser.add_argument('--ld8', type=float, default=0.1)            # 损失4: 提示损失的权重: 节点与聚类中心交叉熵
    # parser.add_argument('--ld9', type=float, default=0.1)            # 损失5: 聚类中心正交性约束损失的权重
    # parser.add_argument('--ld10', type=float, default=0.1)           # 损失6: 高置信度样本伪标签自监督损失的权重

    # 不需要改的
    parser.add_argument('--lr', type=float, default=1e-3)                   # 学习率
    parser.add_argument('--pretrain_path', type=str, default='pkl')         # 预训练模型文件的路径，pkl：序列化格式
    parser.add_argument('--seed', default=123, type=int)                    # 随机数种子
    parser.add_argument('--v', default=1.0, type=float)                     # 调节分配概率的平滑程度
    parser.add_argument('--epoch', default=200, type=int)                   # 训练轮数

    # 解析超参数
    args = parser.parse_args()

    # 添加参数：检查 CUDA 是否可用，并将结果存储在 args.cuda 中
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:3" if args.cuda else "cpu")
    print("use cuda: {}".format(args.cuda))

    # 根据 CUDA 的可用性来选择设备
    device = torch.device("cuda:3" if args.cuda else "cpu")

    # 添加参数：预训练模型文件的路径，pkl：序列化格式
    args.pretrain_path = 'data/pkl/{}.pkl'.format(args.name)

    # 添加超参数：根据不同的数据集设置不同的参数，设置随机数种子
    initial()                   # 初始化：根据不同的数据集设置不同的参数
    # seed_torch(args.seed)     # 设置随机数种子
    print(args)                 # 输出超参数

    '''2. 加载节点数据：特征数据，标签数据'''
    dataset = load_data(args.name)

    '''3. 训练'''
    train_AGCN(dataset)

