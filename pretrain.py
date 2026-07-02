# 开发者：李永桢
# 开发时间：2024/8/7 7:34 PM
# 代码功能：预训练自编码器（AE）

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation_AE import eva

# 确保使用正确的GPU设备（如果有的话）
# torch.cuda.set_device(3)

'''1. 模型：自动编码器（Autoencoder）'''
class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        """
        1.编码器：包含三个线性层（enc_1、enc_2、enc_3）和对应的激活函数（F.relu）。
          最后一个线性层（z_layer）的输出是一个向量，代表了输入数据的压缩表示或编码。

        2.解码器包含三个线性层（dec_1、dec_2、dec_3）和对应的激活函数（F.relu）。
          最后一个线性层（x_bar_layer）的输出是重构的输入数据，它应该尽可能接近原始输入数据。

        3.自动编码器的编码器和解码器的最后一个线性层通常不使用激活函数。这样，模型的输出可以直接用于进一步的处理；
          如果加了激活函数，输出就不是线性的了。
        """
        super(AE, self).__init__()
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
        self.z_layer = nn.Linear(n_enc_3, n_z)

        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = nn.Linear(n_dec_3, n_input)

    def forward(self, x):
        """
        前向传播方法
        :参数 x: 输入的特征矩阵
        :返回值: x_bar是解码器的输出；z是编码器的输出；
        """
        '''1. 编码过程'''
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        '''2. 解码过程'''
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


'''2. 数据集类：加载自定义的数据集'''
class LoadDataset(Dataset):
    def __init__(self, features, labels):
        """
        :方法 __init__: 初始化数据集
        :参数 features: 输入特征，通常是一个二维数组或矩阵
        :参数 labels:   对应的标签，通常是一个一维数组
        """
        self.x = features
        self.y = labels

    def __len__(self):
        """
        :方法 __len__: 定义数据集的长度
        :返回值：       self.x 的行数，即样本的数量
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        :方法 __getitem__:   允许通过索引 idx 访问数据集中的单个样本
        :参数 idx:           索引为 idx
        :返回值 self.x[idx]: 将索引为 idx 的样本从NumPy数组转换为PyTorch的Tensor，并且指定数据类型为浮点数（.float()）
        :返回值 idx:         索引
        """
        return torch.from_numpy(self.x[idx]).float(), \
            torch.from_numpy(np.array(idx))


'''3. 调整学习率：根据当前的训练 epoch（训练轮数）来调整优化器的学习率'''
def adjust_learning_rate(optimizer, epoch):
    """
    根据当前的训练 epoch（训练轮数）来调整优化器的学习率
    :参数 optimizer: 优化器
    :参数 epoch:     训练轮次
    """

    '''
    1.计算新的学习率：初始学习率设置为 0.001；epoch // 20 是一个整除操作，它每20个epoch将学习率减小10倍（即乘以0.1）；
      这意味着在第20个epoch时，学习率将变为 0.001 * 0.1 = 0.0001，在第40个epoch时，学习率将变为 0.0001 * 0.1 = 0.00001，以此类推
    '''
    lr = 0.001 * (0.1 ** (epoch // 20))

    '''2. 遍历优化器中的每个参数组（param_group）。一个优化器可以有多个参数组，每个参数组可以有自己的学习率和其他设置'''
    for param_group in optimizer.param_groups:
        # 将计算出的新学习率 lr 赋值给当前参数组的 ‘lr’ 键，从而更新优化器中所有参数组的学习率
        param_group['lr'] = lr


'''4. 预训练方法：预训练自编码器'''
def pretrain_ae(model, dataset, n_clusters, name, Epoch):
    """
    :方法 pretrain_ae: 预训练自编码器
    :参数 model:       要预训练的模型
    :参数 dataset:     包含数据的数据集对象
    :返回值:
    """

    '''1. 创建了一个 DataLoader 对象 train_loader，用于在训练过程中加载数据：batch_size=256 指定了每个批次加载的样本数量'''
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    optimizer = Adam(model.parameters(), lr=1e-3)                           # 优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 设备
    model.to(device)                                                        # 把模型移动到设备上

    '''2. 训练自编码器模型：轮，每轮整个数据集会被完整遍历一次'''
    for epoch in range(Epoch):
        '''
        1.每个 epoch 调整优化器的学习率：
          optimizer 是一个可变对象，它包含模型参数和学习率等属性。
          当你在函数内部修改 optimizer 的学习率时，实际上是在修改 optimizer 对象本身，而不是它的副本。
          因此，函数内部所做的更改会保留在函数外部。
        '''
        adjust_learning_rate(optimizer, epoch)

        '''
        2.使用 enumerate 函数遍历 train_loader 返回的批次数据：
          batch_idx 是批次的索引。
          (x, _) 是从 train_loader 中获取的一个批次的数据，其中 x 是特征数据，_ 是索引（在这里没有使用，所以用 _ 表示）。
        '''
        for batch_idx, (x, _) in enumerate(train_loader):

            x = x.to(device)             # 将批次数据 x 移动到之前确定的设备上（GPU或CPU）

            x_bar, z = model(x)          # 通过模型 model 进行前向传播，得到重构的特征 x_bar 和编码器输出 z
            loss = F.mse_loss(x_bar, x)  # 计算重构损失，这里使用均方误差（MSE）作为损失函数，比较重构的特征 x_bar 和原始特征 x

            optimizer.zero_grad()  # 梯度清零
            loss.backward()        # 反向传播：计算当前批次损失关于模型参数的梯度
            optimizer.step()       # 参数更新：根据计算出的梯度更新模型参数

        '''
        3.评估模型：使用 torch.no_grad() 上下文管理器，在这个上下文中，所有涉及到的Tensor不会计算梯度
        '''
        with torch.no_grad():

            # 将整个数据集的特征 dataset.x 转换为 PyTorch 的 Tensor，移动到之前确定的设备上（GPU或CPU），并转换为浮点数类型
            x = torch.from_numpy(dataset.x).to(device).float()

            # 对整个数据集进行前向传播，得到重构的特征 x_bar 和编码器输出 z
            x_bar, z = model(x)

            # 计算整个数据集的重构损失，使用均方误差（MSE）作为损失函数
            loss = F.mse_loss(x_bar, x)
            print(f'Epoch {epoch} loss: {loss}')

            # 将编码器输出 z 转移到 CPU 上，并将其转换为 NumPy 数组，以便进行后续的非 PyTorch 操作
            z_np = z.data.cpu().numpy()

            '''
            使用 KMeans 聚类算法对潜在表示 z_np 进行聚类
            这里指定了 3 个聚类中心（n_clusters=3）和 20 次初始质心选择（n_init=20），以获得更好的聚类结果
            '''
            kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(z_np)

            # 调用 eva 函数：传入真实标签 dataset.y、KMeans聚类得到的标签 kmeans.labels_ 和当前epoch的编号，以评估聚类性能。
            eva(dataset.y, kmeans.labels_, epoch)

    '''3. 训练结束后保存模型'''
    torch.save(model.state_dict(), 'data/pkl/{}.pkl'.format(name))




'''主函数'''
if __name__ == "__main__":

    '''1. 创建命令行解析器'''
    parser = argparse.ArgumentParser(
        # 描述，当命令行工具使用--h或--help选项时，这个描述会显示在帮助信息的开始部分，告诉用户这个脚本的用途
        description='train',
        # 一个帮助消息格式化类，它会为每个参数显示其默认值
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 添加超参数
    parser.add_argument('--name', type=str, default='facebook')     # 数据集名字（要改）
    parser.add_argument('--k', type=int, default=None)              # k 近邻的取值
    parser.add_argument('--n_input', default=16, type=int)          # 自编码器的 编码器 输入特征维度（要改）
    parser.add_argument('--n_clusters', default=7, type=int)        # 聚类的数目（要改）
    parser.add_argument('--n_z', default=10, type=int)              # 自编码器的 编码器 输出特征维度
    parser.add_argument('--epoch', default=30, type=int)            # 训练轮数：ACM30; DBLP15(acc 0.5028);

    # 解析超参数
    args = parser.parse_args()

    '''1. 创建模型：自动编码器'''
    # 确定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # n_input=5：输入数据的特征维度；n_z=10：编码器的输出特征维度
    model = AE(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
    ).to(device)

    '''2. np.loadtxt：从文本文件中加载数据，得到特征矩阵x（二维数组）、标签y（一维数组）'''
    x = np.loadtxt('data/{}.txt'.format(args.name), dtype=float)
    y = np.loadtxt('data/{}_label.txt'.format(args.name), dtype=int)

    '''3. 创建数据集对象的实例：dataset 变量现在是一个 LoadDataset 对象的实例，可以使用它来访问数据集中的样本'''
    dataset = LoadDataset(x, y)

    '''4. 预训练自编码器'''
    pretrain_ae(model, dataset, n_clusters=args.n_clusters, name=args.name, Epoch=args.epoch)


