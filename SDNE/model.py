import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class SDNE(nn.Module):
    def __init__(self, node_size, hid_sizes, droput, alpha):
        super(SDNE, self).__init__()
        assert len(hid_sizes) >= 2
        self.num_hids = len(hid_sizes)
        self.emb_size = hid_sizes[-1]
        self.encode1 = nn.Linear(node_size, hid_sizes[0])
        # setattr(self, "fc%d" % i, fc)
        for i in range(1, len(hid_sizes)):
            setattr(self, 'encode{}'.format(i + 1), nn.Linear(hid_sizes[i - 1], hid_sizes[i]))
        for i in range(len(hid_sizes) - 1):
            setattr(self, 'decode{}'.format(i + 1),
                    nn.Linear(hid_sizes[self.num_hids - 1 - i], hid_sizes[self.num_hids - 2 - i]))
        setattr(self, 'decode{}'.format(self.num_hids), nn.Linear(hid_sizes[0], node_size))
        # self.encode1 = nn.Linear(nhid0, nhid1)
        # self.decode0 = nn.Linear(nhid1, nhid0)
        # self.decode1 = nn.Linear(nhid0, node_size)
        self.droput = droput
        self.alpha = alpha

    def forward(self, adj_batch, adj_mat, b_mat):
        t0 = F.leaky_relu(self.encode0(adj_batch))
        for i in range(1, self.num_hids + 1):
            t0 = getattr(self, 'encode{}'.format(i))(t0)
            t0 = F.leaky_relu(t0)
        embedding = t0
        for i in range(self.num_hids + 1):
            t0 = getattr(self, 'decode{}'.format(i))(t0)
            t0 = F.leaky_relu(t0)
        # t0 = F.leaky_relu(self.encode1(t0))
        # embedding = t0
        # t0 = F.leaky_relu(self.decode0(t0))
        # t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # (batch_size,1)
        L_1st = torch.sum(adj_mat * (embedding_norm -
                                     2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                                     + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))
        return L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd

    def savector(self, adj):
        t0 = self.encode0(adj)
        t0 = self.encode1(t0)
        return t0
