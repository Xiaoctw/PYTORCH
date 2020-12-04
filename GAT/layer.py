import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha  # 学习因子
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # 建立都是0的矩阵，大小为（输入维度，输出维度）
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))  # 见下图
        # print(self.a.shape)  torch.Size([16, 1])

        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        print(input.shape)
        print(self.W.shape)
        h = torch.mm(input, self.W)
        # print(h.shape)  torch.Size([2708, 8]) 8是label的个数
        N = h.size()[0]
        # print(N)  2708 nodes的个数

        #计算attention的方便简单的方法
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1,
                                                                                          2 * self.out_features)  # 见下图
        # print(a_input.shape)  torch.Size([2708, 2708, 16])
        # idxs1, idxs2 = [], []
        # 这样添加太慢了
        # for i in range(N):
        #     for j in range(N):
        #         idxs1.append(i)
        #         idxs2.append(j)
        # a_input = torch.cat([h[idxs1], h[idxs2]], dim=1).view(N, N, -1)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # 即论文里的eij
        # squeeze除去维数为1的维度
        # [2708, 2708, 16]与[16, 1]相乘再除去维数为1的维度，故其维度为[2708,2708],与领接矩阵adj的维度一样

        zero_vec = -9e15 * torch.ones_like(e)
        # 维度大小与e相同，所有元素都是-9*10的15次方
        # zero_vec=zero_vec.mul((adj <= 0).int())
        attention = torch.where(adj > 0, e, zero_vec)
        # attention = e.add(zero_vec)
        '''这里我们回想一下在utils.py里adj怎么建成的：两个节点有边，则为1，否则为0。
        故adj的领接矩阵的大小为[2708,2708]。(不熟的自己去复习一下图结构中的领接矩阵)。
        print(adj）这里我们看其中一个adj
        tensor([[0.1667, 0.0000, 0.0000,  ..., 0.0000, 0.0000,   0.0000],
        [0.0000, 0.5000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.2000,  ..., 0.0000, 0.0000, 0.0000],
        ...,
        [0.0000, 0.0000, 0.0000,  ..., 0.2000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.2000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.2500]])
        不是1而是小数是因为进行了归一化处理
        故当adj>0，即两结点有边，则用gat构建的矩阵e，若adj=0,则另其为一个很大的负数，这么做的原因是进行softmax时，这些数就会接近于0了。

        '''
        attention = F.softmax(attention, dim=1)
        # 对应论文公式3，attention就是公式里的αij
        '''print(attention)
        tensor([[0.1661, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.5060, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.2014,  ..., 0.0000, 0.0000, 0.0000],
        ...,
        [0.0000, 0.0000, 0.0000,  ..., 0.1969, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.1998, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.2548]]'''
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        # 如果concat, 说明后面还有层，加上个激活函数，如果没有层那么concat为负值，直接返回
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
