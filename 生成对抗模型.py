import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

num_real_data = 300
batch_size = 1000
lr = 5e-4
N_z = 20
N_data = 2


def generate_real_data():
    # xs = np.linspace(-10, 10, batch_size)
    # data = [[x, np.sin(x)] for x in xs]
    mean=[3,4]
    cov=[[1,0],[0,1]]
    data=np.random.multivariate_normal(mean,cov,batch_size)
    return torch.Tensor(data).float()

# 生产器
G = nn.Sequential(
    nn.Linear(N_z, 40),
    nn.ReLU(),
    nn.Linear(40, N_data)
)
# 判别器
D = nn.Sequential(
    nn.Linear(N_data, 40),
    nn.ReLU(),
    nn.Linear(40, 1),
    nn.Sigmoid(),
)

opt_D = torch.optim.Adam(D.parameters(), lr=lr)
opt_G = torch.optim.Adam(G.parameters(), lr=lr)


def train(max_step):
        loss_es=[]
        data = generate_real_data()
        for step in range(max_step):
            G_input = torch.randn(batch_size, N_z)
            G_output = G(G_input)
            prob1 = D(G_output)
            prob2 = D(data)
            D_loss = -torch.mean(torch.log(prob2)) - torch.mean(torch.log(1 - prob1))
            G_loss = -torch.mean(torch.log(prob1))
            opt_D.zero_grad()
            #保留整个图
            D_loss.backward(retain_graph=True)
            opt_G.zero_grad()
            G_loss.backward()
            opt_D.step()
            opt_G.step()
            loss_es.append(D_loss.item()+G_loss.item())
            if step % 400 == 0:
                plt.scatter(data[:,0],data[:,1],c='y')
                gene_data=G(torch.randn(batch_size,N_z)).data.numpy()
                plt.scatter(gene_data[:,0],gene_data[:,1],c='b')
                plt.show()
        plt.plot(loss_es)
        plt.show()


if __name__ == '__main__':
    train(4000)
