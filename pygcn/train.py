from __future__ import division
from __future__ import print_function
import time
from torch.nn.parameter import Parameter
import argparse
import numpy as np
from itertools import combinations
import torch
import math
import matplotlib.pyplot as plt
from sympy import *
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc,precision_score, recall_score, f1_score, accuracy_score, average_precision_score
import torch.nn as nn
from pygcn.utils import load_data
from pygcn.models import GCN
from pygcn.mutual_information import mutual_information
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings(action='ignore')
start = time.time()
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.7,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--initial_dim', type=int, default=64,
                    help='Dimensions of initial features.')
parser.add_argument('--hidden_dim1', type=int, default=256,
                    help='Dimensions of hidden units.')
# parser.add_argument('--hidden_dim2', type=int, default=128,
#                     help='Dimensions of hidden units.')
parser.add_argument('--output_dim', type=int, default=64,
                    help='Dimensions of output layer.')


args = parser.parse_args()#加载参数
args.cuda = not args.no_cuda and torch.cuda.is_available()# 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True

np.random.seed(args.seed)      #生成随机种子，生成随机数时用
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj_matrix, node_number, order_number_train, order_number_test, neg_train_sample, pos_train_sample, neg_test_sample, pos_test_sample = load_data()
#adj_matrix, adj_matrix_norm, adj_matrix_cir, adj_matrix_cir_norm, lap_matrix, lap_matrix_norm,

# Model and optimizer
model = GCN(nfeat=args.initial_dim,     #初始特征维度
            nhid1=args.hidden_dim1,  #隐藏层的维度，W1的列数
            # nhid2=args.hidden_dim2,
            noutput=args.output_dim,
            dropout=args.dropout,
            node_num=node_number)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1)#

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(adj_matrix)
    print(output)
    NMI_pos, NMI_neg = mutual_information(output, neg_train_sample, pos_train_sample, order_number_train)
    loss_train = cross_loss(NMI_pos, NMI_neg)
    AUC_train, AUC_PR_train, AP_train = AUC(NMI_pos, NMI_neg)
    loss_train.backward()
    optimizer.step()
    print(f'EPOCH[{epoch}/{args.epochs}]')
    print(f"[train_lOSS{epoch}] {loss_train}",
          f"[train_AUC{epoch}] {AUC_train, AUC_PR_train}")
    return loss_train, AUC_train, AUC_PR_train, AP_train


# 定义测试函数，相当于对已有的模型在测试集上运行对应的loss与accuracy
def test():
     model.eval()
     final_feature_matrix = model(adj_matrix)
     tNMI_pos, tNMI_neg = mutual_information(final_feature_matrix, neg_test_sample, pos_test_sample, order_number_test)
     AUC_test, AUC_PR_test, AP_test = AUC(tNMI_pos, tNMI_neg)
     print(f"[test_AUC{epoch}] {AUC_test, AUC_PR_test, AP_test}")


def cross_loss(pos_score, neg_score):
    pos_score = torch.Tensor(pos_score)
    neg_score = torch.Tensor(neg_score)
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    loss1 = F.binary_cross_entropy_with_logits(scores, labels)
    reg_loss = (1 / 2) * ((model.gc1.weight.norm(2)) + (model.gc2.weight.norm(2)))
    reg_loss = reg_loss * args.weight_decay
    # reg_loss = 0
    loss = loss1 + reg_loss
    return loss

def AUC(pos_score, neg_score):
    pos_score = torch.Tensor(pos_score)
    neg_score = torch.Tensor(neg_score)
    scores = torch.cat([pos_score, neg_score]).detach().numpy()  #拼接
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().numpy()
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    AUC_PR = auc(recall, precision)
    return roc_auc_score(labels, scores), AUC_PR, average_precision_score(labels, scores)
 #, tprecision, trecall


# Train model
t_total = time.time()
loss_vector = []
AUC_PR_vector = []
AUC_vector = []
AP_vector = []
idx_train = range(order_number_train*2)
idx_test = range(order_number_train*2, (order_number_train+order_number_test)*2)

for epoch in range(args.epochs):
    loss_train, AUC_train, AUC_PR_train, AP_train = train(epoch)
    loss_vector.append(loss_train.data)
    AUC_PR_vector.append(AUC_PR_train)
    AUC_vector.append(AUC_train)
    AP_vector.append(AP_train)
print('======================')
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

test()

## 画图
fig, x = plt.subplots(1, 1)
epoch_number = list(range(1, args.epochs + 1))
x.plot(epoch_number, loss_vector, color='blue', label='Training loss', linewidth=1.3, linestyle='-', marker='o', markersize=4)
x.plot(epoch_number, AUC_vector, color='green', label='AUC', linewidth=1.3, linestyle='-', marker='d', markersize=4)
x.plot(epoch_number, AUC_PR_vector, color='red', label='AUC_PR', linewidth=1.3, linestyle='-', marker='*', markersize=4)
plt.legend()
plt.show()


