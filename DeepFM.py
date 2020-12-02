import math

import torch
import torch.nn as nn
import torch.nn.functional as fun
import matplotlib.pyplot as plt
import torch.utils.data as Data
from helper import *

'''
处理离散型特征的
DeepFM模型
'''


class DeepFM(nn.Module):
    """
    点击预测模型值deepFM
    """
    def __init__(self, field_size, feature_sizes, embedding_size=4,
                 h_depth=2, deep_layers=None, dropout_shallow=None, dropout_deep=None, task='binary'):
        super(DeepFM, self).__init__()
        # 默认中间有两个连续层，12个节点和8个节点
        self.task = task
        if dropout_deep is None:
            dropout_deep = [0.2, 0.2, 0.2]
        if dropout_shallow is None:
            dropout_shallow = [0.2, 0.2]
        if deep_layers is None:
            deep_layers = [12, 8]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.deep_layers = deep_layers
        self.h_depth = h_depth
        self.emb_size = embedding_size
        self.feat_sizes = feature_sizes
        self.field_size = field_size
        self.dropout_deep = dropout_deep  # 这个是在deep网络部分使用的dropout系数
        self.dropout_shallow = dropout_shallow  # 这个是在一维特征和组合特征上使用的dropout
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)
        stdv = math.sqrt(1.0 / len(self.feat_sizes))
        print('初始化deepFM中FM部分')
        self.dropout1 = nn.Dropout(dropout_shallow[0])
        # 这一部分可以看做是LR
        self.fm_first = nn.Embedding(sum(feature_sizes), 1)
        self.fm_first.weight.data.normal_(0, std=stdv)
        # 交叉连接层
        self.fm_second = nn.Embedding(sum(feature_sizes), self.emb_size)
        self.dropout2 = nn.Dropout(dropout_shallow[1])
        self.fm_second.weight.data.normal_(0, std=stdv)
        print('初始化deepFM中Deep模型')
        # 一个全连接层
        self.lin_1 = nn.Linear(self.field_size * self.emb_size, self.deep_layers[0])
        self.deep_drop_0 = nn.Dropout(self.dropout_deep[0])
        self.batch_norm_1 = nn.BatchNorm1d(self.deep_layers[0])
        self.deep_drop_1 = nn.Dropout(self.dropout_deep[1])
        for i, h in enumerate(self.deep_layers[1:], 1):
            setattr(self, 'lin_' + str(i + 1), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
            setattr(self, 'batch_norm_' + str(i + 1), nn.BatchNorm1d(self.deep_layers[i]))
            setattr(self, 'deep_drop_' + str(i + 1), nn.Dropout(self.dropout_deep[i + 1]))
        # self.dropout3 = nn.Dropout(dropout_shallow[2])
        print('初始化Deep模型完成')

    def forward(self, x):
        num_item = x.shape[0]
        x1 = x.view(num_item * self.field_size)
        fm_first = self.fm_first(x1)
        fm_first = fm_first.view(x.size(0), -1)
        fm_first = self.dropout1(fm_first)
        fm_sec_emb = self.fm_second(x1).view(x.size(0), self.field_size, -1)  # (batch_size,field_size,embedding_size)
        #  print('fm_sec_emb:{}'.format(fm_sec_emb.shape))
        fm_sum_sec_emb = torch.sum(fm_sec_emb, 1)  # (batch_size,embedding_size)
        #  print('fm_sum_Sec_emb{}'.format(fm_sum_sec_emb.shape))
        # (batch_size,embedding_size)
        fm_sum_sec_emb_squ = fm_sum_sec_emb * fm_sum_sec_emb  # (x+y)^2
        # (batch_size,field_size,embedding_size)
        fm_sec_emb_squ = fm_sec_emb * fm_sec_emb
        # (batch_size,embedding_size)
        fm_sec_emb_squ_sum = torch.sum(fm_sec_emb_squ, 1)  # x^2+y^2
        fm_second = (fm_sum_sec_emb_squ - fm_sec_emb_squ_sum) * 0.5
        # (batch_size,embedding_size)
        fm_second = self.dropout2(fm_second)
        deep_emb = fm_sec_emb.reshape(num_item, -1)
        deep_emb = self.deep_drop_0(deep_emb)
        x_deep = fun.relu(self.batch_norm_1(self.lin_1(deep_emb)))
        x_deep = self.deep_drop_1(x_deep)
        for i in range(1, len(self.deep_layers)):
            x_deep = getattr(self, 'lin_' + str(i + 1))(x_deep)
            x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
            x_deep = fun.relu(x_deep)
            x_deep = getattr(self, 'deep_drop_' + str(i + 1))(x_deep)
        # 返回总的结果
        if self.task == 'binary':
            total_sum = torch.sigmoid(
                torch.sum(fm_first, 1) + torch.sum(fm_second, 1) + torch.sum(x_deep, 1) + self.bias)
        else:
            total_sum = torch.sum(fm_first, 1) + torch.sum(fm_second, 1) + torch.sum(x_deep, 1) + self.bias
        return total_sum


if __name__ == '__main__':
    x = torch.randint(1, 4, (3000, 8))
    x, field_size, feature_sizes = find_deep_params(x)
    x = torch.Tensor(x).long()
    y = torch.randint(0, 2, (3000,)).float()
    # print(y[:40])
    deepFM = DeepFM(field_size=field_size, feature_sizes=feature_sizes)
    train(deepFM, x, y, num_epoch=200, lr=3e-2)
    print(deepFM(x)[:10])
    print(y[:10])
