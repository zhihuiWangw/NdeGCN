import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
# from train import args
import numpy as np
import math

class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, noutput, dropout, node_num):#nhid2,
        super(GCN, self).__init__()  #super()._init_()利用父类里的对象构造函数

        self.gc1 = GraphConvolution(nfeat, nhid1)      #第一层的输入输出，输出维度：64，64
        self.gc2 = GraphConvolution(nhid1, noutput)
        # self.gc2 = GraphConvolution(nhid1, nhid2)    #第二层的输入输出，输出维度：64，64
        # self.gc3 = GraphConvolution(nhid2, noutput)
        self.initial_state_matrix = torch.nn.Embedding(node_num, nfeat).double()
        self.initial_state_matrix.weight.data = math.pi / 2 * torch.rand(node_num, nfeat).double()#
        # x = math.pi / 2 * torch.rand(node_num, nfeat).double()
        self.dropout = dropout

        self.sm = nn.Sigmoid()

    def forward(self, adj_matrix):
        x = self.initial_state_matrix.weight
        x1 = F.relu(self.gc1(x, adj_matrix))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = self.gc2(x1, adj_matrix)  # F.relu
        # x1 = F.dropout(x1, self.dropout, training=self.training)
        # x1 = self.gc3(x1, adj_matrix)
        # x = x + x1
        #return F.log_softmax(x, dim=1)
        return self.sm(x1)
        #return x self.sm

    # def forward(self, matrix, incidence_matrix):
    #     # a0 = 1
    #     # a1 = 1/2
    #     # a2 = 1/3
    #     # a3 = 1/4
    #     x0 = self.initial_state_matrix.weight
    #     x1 = self.gc1(x0, matrix, incidence_matrix)
    #     x1 = F.dropout(x1, self.dropout, training=self.training)
    #     x2 = self.gc2(x1, matrix, incidence_matrix)
    #     x2 = F.dropout(x2, self.dropout, training=self.training)
    #     x3 = self.gc3(x2, matrix, incidence_matrix)  # F.relu
    #     # xx = a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3
    #     # return F.log_softmax(x, dim=1)
    #     return (x3)
        # return x
    # 激活函数
    """
    sigmoid   [0,1]           self.sm
    tanh      [-1,1]          F.tanh
    Relu      max(0,x)        F.relu
    leakyRelu max(0.01x,x)    F.leaky_relu
    softmax   [0,1]           F.softmax
    sin       [-1,1]          torch.sin
    Softplus  [0,x]           F.Softplus
    """