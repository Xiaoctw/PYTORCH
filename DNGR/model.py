#
# 计算PPMI矩阵的MATLAB方法，列向量乘行向量会得到一个矩阵
#
# function PPMI = GetPPMIMatrix(M)
#
# M = ScaleSimMat(M);
#
# [p, q] = size(M);
# assert(p==q, 'M must be a square matrix!');
#
# col = sum(M);
# row = sum(M,2);
#
# D = sum(col);
# PPMI = log(D * M ./(row*col));
# PPMI(PPMI<0)=0;
# IdxNan = isnan(PPMI);
# PPMI(IdxNan) = 0;
#
# end
import warnings
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch


class DataSet(Data.Dataset):

    def __init__(self, mat):
        self.Adj = mat
        self.num_node = self.Adj.shape[0]

    def __getitem__(self, index):
        return self.Adj[index]

    def __len__(self):
        return self.num_node


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = np.diag(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx


def random_surfing(adj: np.ndarray, epochs: int, alpha: float) -> np.ndarray:
    """
    :param adj: 邻接矩阵，numpy数组
    :param epochs: 最大迭代次数
    :param alpha: random surf 过程继续的概率
    :return: numpy数组
    """
    N = adj.shape[0]
    adj = normalize(adj)
    P0, P = np.eye(N), np.eye(N)
    mat = np.zeros((N, N))
    for _ in range(epochs):
        P = alpha * P.dot(adj) + (1 - alpha) * P0
        mat = mat + P
    return mat


def PPMI_matrix(mat: np.ndarray) -> np.ndarray:
    """
    :param mat: 上一步构建完成的corjuzhen
    """
    m, n = mat.shape
    assert m == n
    D = np.sum(mat)
    col_sums = np.sum(mat, axis=0)
    row_sums = np.sum(mat, axis=1).reshape(-1, 1)
    dot_mat = row_sums * col_sums
    PPMI = np.log(D * mat / dot_mat)
    PPMI = np.maximum(PPMI, 0)
    PPMI[np.isinf(PPMI)] = 0
    #  PPMI = PPMI / PPMI.sum(1).reshape(-1, 1)
    return PPMI


class AutoEncoderLayer(nn.Module):
    """
    堆叠的自编码器中的单一一层
    """
    def __init__(self, input_dim, output_dim, zero_ratio, GPU):
        super(AutoEncoderLayer, self).__init__()
        self.zero_ratio = zero_ratio
        self.GPU = GPU
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, zero=True):
        if not self.GPU:
            if zero:
                x = x.cpu().clone()
                rand_mat = torch.rand(x.shape)
                zero_mat = torch.zeros(x.shape)
                # 随机初始化为0
                x = torch.where(rand_mat > self.zero_ratio, x, zero_mat)
        else:
            if zero:
                x = x.clone()
                rand_mat = torch.rand(x.shape, device='cuda')
                zero_mat = torch.zeros(x.shape, device='cuda')
                # 随机初始化为0
                x = torch.where(rand_mat > self.zero_ratio, x, zero_mat)
        x = self.encoder(x)
        #   x = F.leaky_relu(x, negative_slope=0.2)
        x = self.decoder(x)
        return x

    def emb(self, x):
        """
        获得该层的嵌入向量
        """
        x = self.encoder(x)
        return x


class StackAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, zero_ratio, GPU=False):
        super(StackAutoEncoder, self).__init__()
        assert len(hidden_dims) >= 1
        self.num_layers = len(hidden_dims) + 1
        self.zero_ratio = zero_ratio
        self.GPU = GPU
        setattr(self, 'autoEncoder0', AutoEncoderLayer(input_dim, hidden_dims[0], zero_ratio=zero_ratio, GPU=GPU))
        for i in range(1, len(hidden_dims)):
            setattr(self, 'autoEncoder{}'.format(i),
                    AutoEncoderLayer(hidden_dims[i - 1], hidden_dims[i], zero_ratio=zero_ratio, GPU=GPU))
        setattr(self, 'autoEncoder{}'.format(self.num_layers - 1),
                AutoEncoderLayer(hidden_dims[-1], output_dim, zero_ratio=zero_ratio, GPU=GPU))
        self.init_weights()

    def emb(self, x):
        for i in range(self.num_layers):
            x = getattr(self, 'autoEncoder{}'.format(i)).emb(x)
        return x

    def forward(self, x):
        # for i in range(self.num_layers):
        #     x = getattr(self, 'autoEncoder{}'.format(i))(x, False)
        for i in range(self.num_layers):
            x = getattr(self, 'autoEncoder{}'.format(i)).encoder(x)
        #      x=F.leaky_relu(x,negative_slope=0.2)
        for i in range(self.num_layers - 1, -1, -1):
            x = getattr(self, 'autoEncoder{}'.format(i)).decoder(x)
        #      x = F.leaky_relu(x, negative_slope=0.2)
        return x

    def init_weights(self):
        # 初始化参数十分重要，可以显著降低loss值
        # 初始化模型参数
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                #mean=0,std = gain * sqrt(2/fan_in + fan_out)
                nn.init.xavier_uniform_(m.weight,gain=1)
            if isinstance(m, nn.BatchNorm1d):
                # nn.init.constant(m.weight, 1)
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # nn.init.constant(m.bias, 0)


if __name__ == '__main__':
    # dataset = 'cora'
    # path = Path(__file__).parent / 'data'
    # print('Loading {} dataset...'.format(dataset))
    # labels = np.load(path / '{}_labels.npy'.format(dataset))
    # features = sp.load_npz(path / '{}_features.npz'.format(dataset))
    # adj = sp.load_npz(path / '{}_adj.npz'.format(dataset))
    # adj = adj.todense()
    # mat = random_surfing(adj, 5, 0.8)
    # mat = PPMI_matrix(mat)
    # print(np.sum(mat))
    model = StackAutoEncoder(10, [8, 6, 5], 2, 0.2)
    # model.__getattr__('autoEncoder0'
    for par in model.parameters():
        par.requires_grad = False
        print(par)
