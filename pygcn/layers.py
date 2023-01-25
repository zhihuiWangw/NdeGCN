import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):   #初始化了一些用到的参数，包括输入和输出的维度，并初始化了每一层的权重
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        print(self.weight)
        if bias:
            self.bias = Parameter(torch.DoubleTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    # 初始化权重
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_matrix1):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj_matrix1, support)#A~ * X * W(0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

        if self.bias is not None:
            return final_output + self.bias
        else:
            return final_output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

  # torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，torch.spmm(a,b)是稀疏矩阵相乘

# def update_weight(P):
#     W1 = model.gc1.weight.ravel()
#     W2 = model.gc2.weight.ravel()
#     W = torch.cat([W1, W2], dim=0)
#     final_W = torch.zeros(W.shape[0])
#     for i in range(P.shape[0]):
#         mean_score = torch.mean(P[i, :])
#         bigger = np.where(P[i, :] >= mean_score)
#         a = torch.dot(torch.squeeze(P[i, bigger], 0), torch.sin(W[i] - W[bigger]))
#         smaller = np.where(P[i, :] < mean_score)
#         b = torch.dot(torch.squeeze(P[i, smaller], 0), torch.sin(W[i] - W[smaller]))
#         final_W[i] = alpha * W[i] - 0.5 * beta * (W[i] ** 2) + K1 * a + K2 * b
#     final_W1 = final_W[0: W1.shape[0]]
#     final_W2 = final_W[W1.shape[0]: final_W.shape[0]]
#     W1 = torch.chunk(final_W1, args.initial_dim, dim=-1)
#     W1 = torch.stack(W1).double()
#     W2 = torch.chunk(final_W2, args.hidden_dim1, dim=-1)
#     W2 = torch.stack(W2).double()
#     return W1, W2