'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run CCLAFMB.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')
    parser.add_argument('--dataset', nargs='?', default='Beibei',
                        help='Choose a dataset from {Beibei,Taobao}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--is_norm', type=int, default=1,
                    help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=128,
                        help='Embedding size. 128 for Beibei,256 for Taobao')
    parser.add_argument('--layer_size', nargs='?', default='[128,128,128]',
                        help='Output sizes of every layer, [128,128,128] for Beibei, [256,256,256] for Taobao')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--model_type', nargs='?', default='CCLAFMB',
                        help='The name of model.')
    parser.add_argument('--adj_type', nargs='?', default='pre',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Gpu id')
    parser.add_argument('--node_dropout_flag', type=int, default=1,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='K eep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--Ks', nargs='?', default='[10, 50, 100]',
                        help='K for Top-K list')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--test_flag', nargs='?', default='full',
                        help='Specify the test type from {part, full}, ither the reference is done inndicating whe mini-batch')

    parser.add_argument('--wid', nargs='?', default='[0.1,0.1,0.1]',
                        help='negative weight, [0.1,0.1,0.1] for beibei, [0.01,0.01,0.01] for taobao')
    parser.add_argument('--decay', type=float, default=10,
                        help='Regularization, 10 for beibei, 0.01 for taobao')
    parser.add_argument('--decay_cl', type=float, default=0.01, help='Regularization, 10 for beibei, 0.01 for taobao')
    parser.add_argument('--coefficient', nargs='?', default='[0.05, 0.80, 0.15]',
                        help='Regularization, [0.05, 0.80, 0.15] for beibei, [0.15, 0.50, 0.35] for taobao')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.2]',
                        help='Keep probability w.r.t. message dropout')




    return parser.parse_args()
