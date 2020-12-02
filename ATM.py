import math

import torch
import torch.nn as nn

from helper import *


class AFM(nn.Module):
    def __init__(self, field_size, feat_sizes, embedding_size=4, task='binary'):
        super(AFM, self).__init__()
        self.task = task
        self.field_size = field_size
        self.feature_sizes = feat_sizes
        self.embedding_size = embedding_size
        self.A = field_size * (field_size - 1) // 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        stdv = math.sqrt(1 / len(self.feature_sizes))
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)
        self.fm_first = nn.Embedding(sum(feat_sizes), 1)
        self.fm_first.weight.data.normal_(0, std=stdv)
        self.fm_second = nn.Embedding(sum(feat_sizes), embedding_size)
        self.W = nn.Linear(self.embedding_size, self.A)
        self.b = nn.Parameter(torch.randn(self.A), requires_grad=True)
        self.h = nn.Linear(self.A, 1)
        self.p = nn.Linear(self.embedding_size, 1)

    def forward(self, x):
        '''

        :param x: batch_size*field_size
        :return:
        '''
        batch_size = x.shape[0]
        fm_first = self.fm_first(x).squeeze(2)
        fm_second = self.fm_second(x)  # batch_size*field_size*embedding_size
        # v_list=[]
        rows, cols = [], []
        for i in range(self.field_size):
            for j in range(i + 1, self.field_size):
                rows.append(i)
                cols.append(j)
                # x2=fm_second[:,i,:]*fm_second[:,j,:]#batch_size*embedding_size
                # v_list.append(x2)
        v = fm_second[:, rows, :] * fm_second[:, cols, :]
        #  v=torch.cat(v_list,dim=0).view(batch_size,self.A,-1)# batch_size*A*embedding_size
        weights = self.h(torch.relu(self.W(v) + self.b)).squeeze(2)  # batch_size*A
        weights = weights.softmax(dim=1)  # batch_size*A
        atm = torch.bmm(weights.unsqueeze(1), v).squeeze(1)
        atm = self.p(atm).view(batch_size)
        total_sum = torch.sum(fm_first, 1) + atm + self.bias
        if self.task == 'binary':
            total_sum = torch.sigmoid(total_sum)
        return total_sum


if __name__ == '__main__':
    x = torch.randint(1, 4, (3000, 8))
    x, field_size, feature_sizes = find_deep_params(x)
    x = torch.Tensor(x).long()
    y = torch.randint(0, 1, (3000,)).float()
    # print(y[:40])
    atm = AFM(field_size=field_size, feat_sizes=feature_sizes)
    train(atm, x, y, num_epoch=200, lr=3e-2)
    print(atm(x)[:10])
    print(y[:10])
