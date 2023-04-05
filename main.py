import argparse
import warnings
from data_loader import MKGCNDataLoader
from train import train
from utils import args_print

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()

# music-d
parser.add_argument( '--dataset', type=str, default='music-d', help='which dataset to use' )
parser.add_argument( '--n_epochs', type=int, default=20, help='the number of epochs' )
parser.add_argument( '--n_neighbor', type=int, default=8, help='the size of neighbors to be sampled' )
parser.add_argument( '--dim', type=int, default=16, help='dimension of user and entity embeddings' )
parser.add_argument( '--n_iter', type=int, default=2, help='number of iterations when computing entity representation' )
parser.add_argument( '--batch_size', type=int, default=64, help='batch size' )
parser.add_argument( '--l2_weight', type=float, default=1e-4, help='weight of l2 regularization' )
parser.add_argument( '--lr', type=float, default=5e-4, help='learning rate' )
parser.add_argument( '--ratio', type=float, default=0.8, help='size of training dataset' )
parser.add_argument( '--sampling_strategy', type=str, default='proportional', help='which negative sampling strategy to use' )

parser.add_argument( '--multimodal', type=bool, default=True, help='whether use multimodal mode')
# lyrics_word2vec 0; mfcc_bow 1; chroma_bow 2; emobase_bow 3; essentia 4; lyrics_sentiment 5; resnet 6.
parser.add_argument( '--modals', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6], help='which modals to use')

parser.add_argument( '--user_aggregator', type=str, default='multi-head', help='which user-item aggregator to use' )
parser.add_argument( '--neighbor_aggregator', type=str, default='sum', help='which item-entity aggregator to use' )
parser.add_argument( '--multimodal_aggregator', type=str, default='residual', help='which item-multimodal aggregator to use' )

parser.add_argument( '--show_topk', type=bool, default=False, help='whether to show precision, recall and ndcg' )
parser.add_argument( '--show_loss', type=bool, default=False, help='whether to show loss' )
parser.add_argument( '--gpu', type=int, default=0, help='which gpu to use' )

args = parser.parse_args()

args_print(args)

data_loader = MKGCNDataLoader(args)
train(args, data_loader)