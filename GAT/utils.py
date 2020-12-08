import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path

'''
先将所有由字符串表示的标签数组用set保存，set的重要特征就是元素没有重复，
因此表示成set后可以直接得到所有标签的总数，随后为每个标签分配一个编号，创建一个单位矩阵，
单位矩阵的每一行对应一个one-hot向量，也就是np.identity(len(classes))[i, :]，
再将每个数据对应的标签表示成的one-hot向量，类型为numpy数组
'''
def encode_onehot(labels):
    classes = set(labels)  # set() 函数创建一个无序不重复元素集
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in  # identity创建方矩阵
                    enumerate(classes)}     # 字典 key为label的值，value为矩阵的每一行
    # enumerate函数用于将一个可遍历的数据对象组合为一个索引序列
    labels_onehot = np.array(list(map(classes_dict.get, labels)),  # get函数得到字典key对应的value
                             dtype=np.int32)
    return labels_onehot
    # map() 会根据提供的函数对指定序列做映射
    # 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表
    #  map(lambda x: x ** 2, [1, 2, 3, 4, 5])
    #  output:[1, 4, 9, 16, 25]


def load_data(dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    path='./'+dataset+'/'
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 储存为csr型稀疏矩阵
    labels = encode_onehot(idx_features_labels[:, -1])
    # content file的每一行的格式为 ： <paper_id> <word_attributes>+ <class_label>
    #    分别对应 0, 1:-1, -1
    # feature为第二列到倒数第二列，labels为最后一列

    # build graph
    # cites file的每一行格式为：  <cited paper ID>  <citing paper ID>
    # 根据前面的contents与这里的cites创建图，算出edges矩阵与adj 矩阵
    idx = np.array(idx_features_labels[:, 0], dtype=str)
    idx_map = {j: i for i, j in enumerate(idx)}
    # 由于文件中节点并非是按顺序排列的，因此建立一个编号为0-(node_size-1)的哈希表idx_map，
    # 哈希表中每一项为id: number，即节点id对应的编号为number
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=str)
    # edges_unordered为直接从边表文件中直接读取的结果，是一个(edge_num, 2)的数组，每一行表示一条边两个端点的idx
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),  # flatten：降维，返回一维数组
    #                  dtype=np.int).reshape(edges_unordered.shape)
    edges=[]
    for i in range(len(edges_unordered)):
        idx1=idx_map.get(edges_unordered[i][0])
        idx2=idx_map.get(edges_unordered[i][1])
        if idx1 is not None and idx2 is not None:
            edges.append([idx1,idx2])
    edges=np.array(edges)
    # 边的edges_unordered中存储的是端点id，要将每一项的id换成编号。
    # 在idx_map中以idx作为键查找得到对应节点的编号，reshape成与edges_unordered形状一样的数组
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),  # coo型稀疏矩阵
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # 根据coo矩阵性质，这一段的作用就是，网络有多少条边，邻接矩阵就有多少个1，
    # 所以先创建一个长度为edge_num的全1数组，每个1的填充位置就是一条边中两个端点的编号，
    # 即edges[:, 0], edges[:, 1]，矩阵的形状为(node_size, node_size)。


    # build symmetric adjacency matrix   论文里A^=(D~)^0.5 A~ (D~)^0.5这个公式
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    sp.save_npz('{}_adj.npz'.format(dataset), adj)
    sp.save_npz('{}_features.npz'.format(dataset), features)
    # 对于无向图，邻接矩阵是对称的。上一步得到的adj是按有向图构建的，转换成无向图的邻接矩阵需要扩充成对称矩阵
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))   # eye创建单位矩阵，第一个参数为行数，第二个为列数
    # 对应公式A~=A+IN

    # 分别构建训练集、验证集、测试集，并创建特征矩阵、标签向量和邻接矩阵的tensor，用来做模型的输入
    features = torch.FloatTensor(np.array(features.todense()))  # tensor为pytorch常用的数据结构
    labels = torch.LongTensor(np.where(labels)[1])
    np.save('{}_labels.npy'.format(dataset), labels)
    adj = torch.FloatTensor(adj.todense()) # 邻接矩阵转为tensor处理
    return adj, features, labels

def load_prepared_data(dataset='cora'):
    # sp.save_npz('{}_adj.npz'.format(dataset), adj)
    # sp.save_npz('{}_features.npz'.format(dataset), features)
    # # sp.save_npz('{}_labels.npz'.format(dataset),save_labels)
    # np.save('{}_labels.npy'.format(dataset), save_labels)
    path=Path(__file__).parent/'data'
    labels = np.load(path/'{}_labels.npy'.format(dataset))
    features = sp.load_npz(path/'{}_features.npz'.format(dataset))
    adj = sp.load_npz(path/'{}_adj.npz'.format(dataset))
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))  # eye创建单位矩阵，第一个参数为行数，第二个为列数
    adj = normalize(adj)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = torch.FloatTensor(adj.todense())
    return adj, features, labels

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels) # 使用type_as(tesnor)将张量转换为给定类型的张量。
    correct = preds.eq(labels).double()  # 记录等于preds的label eq:equal
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):    # 把一个sparse matrix转为torch稀疏张量
    """
    numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    pytorch中的tensor转化成numpy中的ndarray : numpy()
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # 不懂的可以去看看COO性稀疏矩阵的结构
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

