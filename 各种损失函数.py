import torch
import torch.nn.functional as F

target_binary = torch.Tensor([0, 1, 1, 0, 0, 1, 0, 0])
target_binary_long = target_binary.long()
b = torch.Tensor([0.3, 0.8, 0.8, 0.2, 0.3, 0.7, 0.3, 0.4])


def binary_cross_entropy(pred, target):
    return -torch.mean(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))


def cross_entropy(pred, target):
    num = target.shape[0]
    arr = pred[range(num), target]
    return -torch.mean(torch.log(arr))


# def nll_loss(pred,target):
#     return -torch.mean()

if __name__ == '__main__':
    print(F.binary_cross_entropy(b.reshape(-1,1), target_binary.reshape(-1,1)))
    print(binary_cross_entropy(b, target_binary))
    c = torch.zeros(b.shape[0], 2)
    c[:, 1] = b
    c[:, 0] = 1 - c[:, 1]
    print(cross_entropy(c, target_binary_long))

# print(c)
