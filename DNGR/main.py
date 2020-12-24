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

parser.add_argument('--lr1', type=float, default=3e-5,
                    help='Train the learning rate at each level')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--epochs2', type=int, default=200)
parser.add_argument('--lr2', type=float, default=3e-5, help='Fine-tuning model learning rate.')
parser.add_argument('--hidden_dims', type=list, default=[1024,256,32])
parser.add_argument('--epoch1s', type=list, default=[200,300,400,500],
                    help='Number of epochs to train.')
parser.add_argument('--output_dim', type=int, default=16)
parser.add_argument('--zero_ratio', type=float, default=0.4, help='The probability of random 0.')
parser.add_argument('--batch_size',type=int,default=512)

args = parser.parse_args()
adj_mat = load_prepared_data()
N = adj_mat.shape[0]
pco_mat = random_surfing(adj_mat, epochs=args.surfing_epoch, alpha=args.alpha)
ppmi_mat = PPMI_matrix(pco_mat)
# 转化为tensor
ppmi_mat = torch.from_numpy(ppmi_mat).float()
GPU = args.cuda and torch.cuda.is_available()
# torch.set_default_tensor_type(torch.DoubleTensor)
sdae = StackAutoEncoder(input_dim=N, hidden_dims=args.hidden_dims, output_dim=args.output_dim,
                        zero_ratio=args.zero_ratio, GPU=GPU)

if __name__ == '__main__':
    if GPU:
        sdae = sdae.cuda()
        ppmi_mat = ppmi_mat.cuda()
    train(sdae, ppmi_mat, lr1=args.lr1, lr2=args.lr2, epoch1s=args.epoch1s, epochs2=args.epochs2,batch_size=args.batch_size)
    torch.save(sdae, 'sdae.pkl')
    save_embeddings(sdae, ppmi_mat, dataset='cora')
    outputs = getattr(sdae, 'autoEncoder0')(ppmi_mat, False)
    print(outputs[0:2])
    print(ppmi_mat[0:2])
    # train_layer(sdae,0,ppmi_mat,lr=args.lr1,epochs=150,print_every=20,batch_size=args.batch_size)
    # embs=getattr(sdae, 'autoEncoder0').emb(ppmi_mat)
    # outputs = getattr(sdae, 'autoEncoder0')(ppmi_mat, False)
    # print(outputs[:2])
    # print(embs[:2])
    #print(ppmi_mat[:2])


