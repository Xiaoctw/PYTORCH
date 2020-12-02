import numpy as np
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.nn as nn


def find_deep_params(x1):
    '''
    这部分作用是返回各个属性上取值的个数和取值的个数，并且对train_x进行处理
    转化为索引矩阵
    :param x2: 测试数据
    :param x1:训练数据
    :return: 变更后的train_x,一共有多少个特征，每个特征上有多少个取值
    '''
    # print(x1.shape)
    # print(x2.shape)
    x1 = np.array(x1)
    size_1, field_size = x1.shape
    # size_2, field_size = x2.shape
    X = np.zeros((size_1, field_size))
    X[:size_1] = x1
    n, field_size = X.shape
    feat_sizes = []
    # 保存每个属性每个值对应的索引
    dic, cnt = {}, 0
    for i in range(field_size):
        feat_sizes.append(np.unique(X[:, i]).shape[0])
        l = np.unique(X[:, i]).tolist()
        for val in l:
            # 每一列上每个元素都有其对应的索引值
            dic[i, val] = cnt
            cnt += 1
    for j in range(field_size):
        for i in range(n):
            val = X[i][j]
            X[i][j] = dic[j, val]
    return X, field_size, feat_sizes


def train(model, x, y, num_epoch=50, lr=3e-4, print_every=5, plot_every=5):
    '''
    训练过程
    :param model: 模型
    :param x:
    :param y:
    :param num_epoch:
    :param lr:
    :param print_every:
    :param plot_every:
    :return:
    '''
    cri = nn.BCELoss(reduction='sum')
    opt = torch.optim.Adam(lr=lr, params=model.parameters())
    data_set = Data.TensorDataset(x, y)
    data_loader = Data.DataLoader(dataset=data_set, batch_size=x.shape[0] // 5, shuffle=True, )  # num_workers=-1)
    losses = []
    for epoch in range(num_epoch):
        total_loss = 0
        for step, (batch_x, batch_y) in enumerate(data_loader):
            opt.zero_grad()
            outputs = model(batch_x)
            loss = cri(outputs, batch_y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if epoch % print_every == 0:
            print('epoch:{},loss:{:.2f}'.format(epoch, total_loss / x.shape[0]))
        if epoch % plot_every == 0:
            losses.append(total_loss / x.shape[0])
    plt.plot(losses, ls='--', color='r')
    plt.scatter(list(range(len(losses))), losses, color='b')
    plt.show()
