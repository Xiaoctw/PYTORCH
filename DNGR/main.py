from model import *
from train import *
from dataset import *
import argparse
import warnings
import torch

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--surfing_epoch', type=int, default=6)
parser.add_argument('--alpha', type=float, default=0.7, help='The probability of going on to the next jump')
parser.add_argument('--epochs1', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr1', type=float, default=3e-4,
                    help='Train the learning rate at each level')
parser.add_argument('--cuda',type=bool,default=True)
parser.add_argument('--epochs2', type=int, default=100)
parser.add_argument('--lr2', type=float, default=3e-5, help='Fine-tuning model learning rate.')
parser.add_argument('--hidden_dims', type=list, default=[256, 64, 16])
parser.add_argument('--output_dim', type=int, default=8)
parser.add_argument('--zero_ratio', type=float, default=0.4, help='The probability of random 0.')

args = parser.parse_args()
adj_mat = load_prepared_data()
N = adj_mat.shape[0]
pco_mat = random_surfing(adj_mat, epochs=args.surfing_epoch, alpha=args.alpha)
ppmi_mat = PPMI_matrix(pco_mat)
# 转化为tensor
ppmi_mat = torch.from_numpy(ppmi_mat).float()
# torch.set_default_tensor_type(torch.DoubleTensor)
sdae = StackAutoEncoder(input_dim=N, hidden_dims=args.hidden_dims, output_dim=args.output_dim,
                        zero_ratio=args.zero_ratio)
GPU=args.cuda and torch.cuda.is_available()

if __name__ == '__main__':
    if GPU:
        sdae=sdae.cuda()
        ppmi_mat=ppmi_mat.cuda()
    train(sdae, ppmi_mat, lr1=args.lr1, lr2=args.lr2, epochs1=args.epochs1, epochs2=args.epochs2)
