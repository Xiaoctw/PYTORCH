import networkx as nx #画图用的
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
import torch
from pathlib import Path

def Read_graph(file_name):
    edge = np.loadtxt(file_name).astype(np.int32)
    min_node, max_node = edge.min(), edge.max()
    if min_node == 0:
        Node = max_node + 1
    else:
        Node = max_node
    G = nx.Graph()
    Adj = np.zeros([Node, Node], dtype=np.int32)
    for i in range(edge.shape[0]):
        G.add_edge(edge[i][0], edge[i][1])
        if min_node == 0:
            Adj[edge[i][0], edge[i][1]] = 1
            Adj[edge[i][1], edge[i][0]] = 1
        else:
            Adj[edge[i][0] - 1, edge[i][1] - 1] = 1
            Adj[edge[i][1] - 1, edge[i][0] - 1] = 1
    Adj = torch.FloatTensor(Adj)
    return G, Adj, Node

class DataSet(data.Dataset):

    def __init__(self, Adj, Node):
        self.Adj = Adj
        self.Node = Node
    def __getitem__(self, index):
        return index
        # adj_batch = self.Adj[index]
        # adj_mat = adj_batch[index]
        # b_mat = torch.ones_like(adj_batch)
        # b_mat[adj_batch != 0] = self.Beta
        # return adj_batch, adj_mat, b_mat
    def __len__(self):
        return self.Node

def read_label():
    path=Path(__file__).parent/'cora'/'cora_labels.txt'
    labels=np.loadtxt(path)[:,1]
    return labels


if __name__ == '__main__':
    #print(read_label())
    G, Adj, Node = Read_graph('./karate/karate.edgelist')
    Data = DataSet(Adj, Node)
    Test = DataLoader(Data, batch_size=20, shuffle=True)
    for index in Test:
        adj_batch = Adj[index]
        adj_mat = adj_batch[:, index]
        b_mat = torch.ones_like(adj_batch)
        b_mat[adj_batch != 0] = 5
        print(b_mat)