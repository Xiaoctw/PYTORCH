import warnings
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as Data
import torch.optim as optim
from model import *


def train(model: nn.Module, X, lr1=3e-5, lr2=3e-6, batch_size=128, epochs1=200, epochs2=50,
          print_every=20):
    for param in model.parameters():
        param.requires_grad = False  # 首先冻结所有的层
    num_layers=model.num_layers
    for i in range(num_layers):
        train_layer(model, i,X ,lr1, epochs1, print_every=print_every)
    for param in model.parameters():
        param.requires_grad = True  # 首先冻结所有的层
    # 对模型整体进行微调
    dataSet = DataSet(X)
    dataLoader = Data.DataLoader(dataset=dataSet, batch_size=batch_size, shuffle=True, )
    optimizer = optim.Adam(model.parameters(), lr=lr2, weight_decay=5e-4)
    criterion = nn.MSELoss()
    for epoch in range(epochs2):
        loss = 0
        for batch_x in dataLoader:
            output = model(batch_x)
            batch_loss = criterion(batch_x, output)
            optimizer.zero_grad()
            batch_loss.backward()
            loss += batch_loss.item()
            optimizer.step()
        if (epoch + 1) % print_every == 0:
            print('Adjust model parameters, epoch:{}, loss: {}'.format(epoch, loss))
    return model

def train_layer(model, i, x, lr=3e-5, epochs=200, print_every=20, batch_size=128):
    for param in getattr(model, 'autoEncoder{}'.format(i)).parameters():
        param.requires_grad = True
    for j in range(i):
        x = getattr(model, 'autoEncoder{}'.format(j)).emb(x)
    dataSet = DataSet(x)
    dataLoader = Data.DataLoader(dataset=dataSet, batch_size=batch_size, shuffle=True, )
    optimizer = optim.Adam(getattr(model, 'autoEncoder{}'.format(i)).parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        loss = 0
        for batch_x in dataLoader:
            output_x = getattr(model, 'autoEncoder{}'.format(i))(batch_x)
            batch_loss = criterion(batch_x, output_x)
            optimizer.zero_grad()
            batch_loss.backward()  # 反向传播计算参数的梯度
            loss += batch_loss.item()
            optimizer.step()
        if (epoch + 1) % print_every == 0:
            print('Train layer {}, epoch:{}, loss: {}'.format( i, epoch+1, loss))
    for param in getattr(model, 'autoEncoder{}'.format(i)).parameters():
        param.requires_grad = False
