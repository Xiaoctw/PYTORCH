import torch
import torch.optim as optim
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.data.dataloader import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data import dataset
from model import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

plt.switch_backend('agg')


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', default='./data/cora/cora_edgelist.txt',
                        help='Input graph file')
    parser.add_argument('--output', default='./data/cora/Vec.emb',
                        help='Output representation file')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--weighted', action='store_true', default=False,
                        help='Treat graph as weighted')
    parser.add_argument('--epochs', default=200, type=int,
                        help='The training epochs of SDNE')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--alpha', default=1e-2, type=float,
                        help='alhpa is a hyperparameter in SDNE')
    parser.add_argument('--beta', default=5., type=float,
                        help='beta is a hyperparameter in SDNE')
    parser.add_argument('--nu1', default=1e-5, type=float,
                        help='nu1 is a hyperparameter in SDNE')
    parser.add_argument('--nu2', default=1e-4, type=float,
                        help='nu2 is a hyperparameter in SDNE')
    parser.add_argument('--bs', default=100, type=int,
                        help='batch size of SDNE')
    # parser.add_argument('--nhid0', default=1000, type=int,
    #                     help='The first dim')
    # parser.add_argument('--nhid1', default=128, type=int,
    #                     help='The second dim')
    parser.add_argument('--hid_sizes', default=[1000, 512, 256, 128], type=list, help='hidden layers')
    parser.add_argument('--step_size', default=10, type=int,
                        help='The step size for lr')
    parser.add_argument('--gamma', default=0.9, type=int,
                        help='The gamma for lr')
    parser.add_argument('--use_batch', default=True, type=bool)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    G, Adj, Node = dataset.Read_graph(args.input)
    model = SDNE(Node, args.hid_sizes, args.dropout, args.alpha)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)
    Data = dataset.DataSet(Adj, Node)
    Data = DataLoader(Data, batch_size=args.bs, shuffle=True, )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device1 = torch.device('cpu')
    model = model.to(device)
    model.train()
    if args.use_batch:
        for epoch in range(1, args.epochs + 1):
            loss_sum, loss_L1, loss_L2, loss_reg = 0, 0, 0, 0
            for index in Data:
                adj_batch = Adj[index]
                # 控制adj_mat为一个小方阵，为(batch_size,batch_size)
                adj_mat = adj_batch[:, index]
                b_mat = torch.ones_like(adj_batch)
                b_mat[adj_batch != 0] = args.beta
                # 转到GPU
                adj_mat = adj_mat.to(device)
                b_mat = b_mat.to(device)
                adj_batch = adj_batch.to(device)
                opt.zero_grad()
                L_1st, L_2nd, L_all = model(adj_batch, adj_mat, b_mat)
                L_reg = 0
                # 加入正则化项
                for param in model.parameters():
                    L_reg += args.nu1 * torch.sum(torch.abs(param)) + args.nu2 * torch.sum(param * param)
                Loss = L_all + L_reg
                Loss.backward()
                opt.step()
                loss_sum += Loss
                loss_L1 += L_1st
                loss_L2 += L_2nd
                loss_reg += L_reg
            scheduler.step(epoch)
            # print("The lr for epoch %d is %f" %(epoch, scheduler.get_lr()[0]))
            print("loss for epoch %d, loss_sum is %f, loss_L1 is %f, loss_L2 is %f, loss_reg is %f:" % (
                epoch, loss_sum, loss_L1, loss_L2, loss_reg))
            # print("loss_sum is %f" % loss_sum)
            # print("loss_L1 is %f" % loss_L1)
            # print("loss_L2 is %f" % loss_L2)
            # print("loss_reg is %f" % loss_reg)


    else:
        b_mat = torch.ones_like(Adj)
        b_mat[Adj != 0] = args.beta
        Adj = Adj.to(device)
        b_mat = b_mat.to(device)
        for epoch in range(1, args.epochs + 1):
            loss_sum, loss_L1, loss_L2, loss_reg = 0, 0, 0, 0
            opt.zero_grad()
            L_1st, L_2nd, L_all = model(Adj, Adj, b_mat)
            L_reg = 0
            # 加入正则化项
            for param in model.parameters():
                L_reg += args.nu1 * torch.sum(torch.abs(param)) + args.nu2 * torch.sum(param * param)
            Loss = L_all + L_reg
            Loss.backward()
            opt.step()
            loss_sum += Loss
            loss_L1 += L_1st
            loss_L2 += L_2nd
            loss_reg += L_reg
            scheduler.step(epoch)
            # print("The lr for epoch %d is %f" %(epoch, scheduler.get_lr()[0]))
            print("loss for epoch %d, loss_sum is %f, loss_L1 is %f, loss_L2 is %f, loss_reg is %f:" % (
                epoch, loss_sum, loss_L1, loss_L2, loss_reg))
            # print("loss_sum is %f" % loss_sum)
            # print("loss_L1 is %f" % loss_L1)
            # print("loss_L2 is %f" % loss_L2)
            # print("loss_reg is %f" % loss_reg)

    model.eval()
    model = model.cpu()
    embedding = model.savector(Adj)
    labels = dataset.read_label()
    outVec = embedding.detach().numpy()
    embs = TSNE(n_components=2).fit_transform(outVec)
    plt.scatter(embs[:, 0], embs[:, 1], c=labels)
    plt.savefig('embedding.png')
