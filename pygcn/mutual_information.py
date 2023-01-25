import numpy as np
import torch
import math
from collections import Counter
from itertools import combinations
import itertools


def mutual_information(output, neg_sample, pos_sample, edge_number):
    neg_mi_value = []
    pos_mi_value = []
    NMI_neg = []
    NMI_pos =[]
    for i in range(edge_number):
        high_size = len(pos_sample[i])
        pos_vector_list = torch.round(10 * (output[pos_sample[i], :])) / 10  # 三行向量
        neg_vector_list = torch.round(10 * (output[neg_sample[i], :])) / 10
        # pos_vector_list = output[pos_sample[i], :]
        # neg_vector_list = output[neg_sample[i], :]
        train_H = []
        neg_H = []
        list2 = list(range(0, high_size))   #np.arange(high_size)
        for j in range(1, high_size+1):
            entro_2list = list(combinations(list2, j))#list()
            for l in range(len(entro_2list)):      #一些2个的组合
                entro_list = torch.LongTensor(entro_2list[l])       #np.array()
                t_X = torch.index_select(pos_vector_list, 0, entro_list)#两行向量
                t_X = t_X.detach().numpy().tolist()
                t_X= list(zip(*itertools.chain(t_X)))
                n_X = torch.index_select(neg_vector_list, 0, entro_list)
                n_X = n_X.detach().numpy().tolist()
                n_X = list(zip(*itertools.chain(n_X)))
                train_H.append(((-1)**(j-1))*Entropy(t_X))
                neg_H.append(((-1)**(j-1))*Entropy(n_X))
        p_mi = np.sum(train_H)
        n_mi = np.sum(neg_H)
        pos_mi_value.append(p_mi)
        neg_mi_value.append(n_mi)
        NMI_neg.append(high_size*n_mi/np.sum(train_H[0:high_size]))
        NMI_pos.append(high_size*p_mi/np.sum(neg_H[0:high_size]))
    return NMI_pos, NMI_neg


def Entropy(DataList):
    counts = len(DataList)  # 总数量
    counter = Counter(DataList)  # 每个变量出现的次数
    prob = {i[0]: i[1] / counts for i in counter.items()}  # 计算每个变量的 p*log(p)
    H = - sum([i[1] * math.log2(i[1]) for i in prob.items()])  # 计算熵

    return H