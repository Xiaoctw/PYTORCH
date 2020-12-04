from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from pathlib import Path
from utils import load_data, accuracy
from models import GAT
from torch.utils import data
from torch.utils.data import DataLoader
from train import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


class DataSet(data.Dataset):

    def __init__(self, Adj, features):
        self.Adj = Adj
        self.features=features

    def __getitem__(self, index):
        return index
        # adj_batch = self.Adj[index]
        # adj_mat = adj_batch[index]
        # b_mat = torch.ones_like(adj_batch)
        # b_mat[adj_batch != 0] = self.Beta
        # return adj_batch, adj_mat, b_mat

    def __len__(self):
        return features.shape[0]


# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

model = GAT(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=int(labels.max()) + 1,
            dropout=args.dropout,
            nheads=args.nb_heads,
            alpha=args.alpha)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    # features = features.cuda()
    # adj = adj.cuda()
    # labels = labels.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()

def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


def save_embeddings():
    model.eval()
    output = model.savector(features, adj)
    outVec = output.cpu().detach().numpy()
    path = Path(__file__).parent / 'cora' / 'outVec.txt'
    np.savetxt(path, outVec)
    path = Path(__file__).parent / 'cora' / 'labels.txt'
    outLabel = labels.cpu().detach().numpy()
    np.savetxt(path, outLabel)
if __name__ == '__main__':
    t_total = time.time()
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    t = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    data = DataSet(adj, features)
    dataLoader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        loss_=[]
        for idx in dataLoader:
            adj_batch = adj[idx][:, idx]
            #  adj_batch = adj_batch[:, idx]
            features_batch = features[idx]
            labels_batch = labels[idx]
            output_batch = model(adj_batch, features_batch)
            loss_train = F.nll_loss(output_batch, labels_batch)
            loss_train.backward()
            loss_.append(loss_train.item())
            optimizer.step()

        model.eval()
        output_eval = model(features[idx_val])
        acc_val = accuracy(output_eval, labels[idx_val])
        loss_val = F.nll_loss(output_eval, labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(np.mean(loss_)),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item()),
              'time: {:.4f}s'.format(time.time() - t))
        # loss_values.append(train(epoch))
        # torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        # # 把效果最好的模型保存下来
        # if loss_values[-1] < best:
        #     best = loss_values[-1]
        #     best_epoch = epoch
        #     bad_counter = 0
        # else:
        #     bad_counter += 1
        #
        # if bad_counter == args.patience:
        #     break
        #
        # files = glob.glob('*.pkl')
        # for file in files:
        #     epoch_nb = int(file.split('.')[0])
        #     if epoch_nb < best_epoch:
        #         os.remove(file)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
    compute_test()
    save_embeddings()
