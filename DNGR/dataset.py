from pathlib import Path
import scipy.sparse as sp
import numpy as np
import torch

def load_prepared_data(dataset='cora'):
    path = Path(__file__).parent / 'data'
    print('Loading {} dataset...'.format(dataset))
    labels = np.load(path/'{}_labels.npy'.format(dataset))
    adj = sp.load_npz(path/'{}_adj.npz'.format(dataset))
    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = torch.FloatTensor(adj.todense())
    return adj
