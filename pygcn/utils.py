import numpy as np
import scipy.sparse as sp
import torch
import random
from collections import Counter


def load_data(path="../data/DAWN/", dataset="DAWN"):
     # tags-ask-ubuntu,  contact-primary-school,  email-Enron, email-Eu,congress-bills, NDC-classes, NDC-substances
    print('Loading {} dataset...'.format(dataset))
    # node_number = 242  # 节点数contact-primary-school
    # edge_number_pair = 97134  # 二元关系
    # order_number_train = 377 # 训练集的三阶的数量7410
    # order_number_test = 94  # 测试集的三阶的数量1852

    # node_number = 565  #NDC-classes
    # edge_number_pair = 25504
    # order_number_train = 814
    # order_number_test = 204

    # node_number = 1511  # tags-math-sx
    # edge_number_pair = 277510
    # order_number_train = 138896
    # order_number_test = 34724

    # node_number = 2722 # tags_ask_ubuntu
    # edge_number_pair = 74881
    # order_number_train = 35382  # 58515#
    # order_number_test = 8845  # 14629#

    node_number = 2003  #DAWN
    edge_number_pair = 553437
    order_number_train = 46670
    order_number_test = 11667

    # node_number = 72346  #threads-ask-ubuntu
    # edge_number_pair = 90061
    # order_number_train = 7632
    # order_number_test = 1908

    # node_number = 945  # email-Eu
    # edge_number_pair = 173288
    # order_number_train = 14366  # 3950
    # order_number_test = 3591  # 988

    # node_number = 1642  # congress-bills
    # edge_number_pair = 32114
    # order_number_train = 6930
    # order_number_test = 1732

    # node_number = 977  #NDC-substances
    # edge_number_pair = 13037
    # order_number_train = 2035
    # order_number_test = 509

    incidence_matrix_pair = np.zeros(shape=(node_number, edge_number_pair))  # 正常的边
    incidence_matrix_preoder = np.zeros(shape=(node_number, order_number_train))  # 高阶结构
    with open("{}{}.simplicies_2".format(path, dataset), "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        splitted = line.strip().split(" ")
        for j in range(len(splitted)):
            incidence_matrix_pair[int(splitted[j]) - 1][i] = 1

    # 二阶关和预测阶的邻接矩阵和拉普拉斯矩阵
    degree_vector_pair = np.sum(incidence_matrix_pair, axis=1)  # 有多少边包含了这个节点,正常和高阶都包含
    degree_matrix_pair = np.diag(degree_vector_pair)
    adj_matrix_pair = incidence_matrix_pair.dot(incidence_matrix_pair.T) - degree_matrix_pair

    # 导入
    pos_train_sample = np.loadtxt(
            'D:/2022年博二寒假/博士论文/1高阶/GCN_Adam/data/DAWN/four_order/train_pos.txt',
            delimiter=",")  # 读入的时候也需要指定逗号分隔
    pos_test_sample = np.loadtxt(
            'D:/2022年博二寒假/博士论文/1高阶/GCN_Adam/data/DAWN/four_order/test_pos.txt',
            delimiter=",")  # 读入的时候也需要指定逗号分隔
    neg_train_sample = np.loadtxt(
            'D:/2022年博二寒假/博士论文/1高阶/GCN_Adam/data/DAWN/four_order/train_neg.txt',
            delimiter=",")  # 读入的时候也需要指定逗号分隔
    neg_test_sample = np.loadtxt(
            'D:/2022年博二寒假/博士论文/1高阶/GCN_Adam/data/DAWN/four_order/test_neg.txt',
            delimiter=",")  # 读入的时候也需要指定逗号分隔

    adj_matrix_pair = normalize(adj_matrix_pair + np.identity(node_number))
    adj_matrix_pair = torch.tensor(adj_matrix_pair)


    return adj_matrix_pair, node_number, order_number_train, order_number_test, neg_train_sample, pos_train_sample, neg_test_sample, pos_test_sample


    """
    adj_matrix        
    adj_matrix_norm
    adj_matrix_cir
    adj_matrix_cir_norm
    lap_matrix
    lap_matrix_norm
    
    adj_matrix_pre
    adj_matrix_pair
    adj_matrix_2pre 
    adj_matrix_norm_2pre 
    adj_matrix_cir_2pre 
    adj_matrix_cir_2pre_norm 
    lap_matrix_2pre 
    lap_matrix_2pre_norm 
    """


    """
       degree_edge_pair         包含的二阶边数 = 连接的二阶节点数 
       degree_node_pre          包含的高阶边数
       degree_edge_pre          连接的高阶节点数
       cycle_ratio              循环比
    """

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))#属性0,1矩阵的行和，每个样本的度
    r_inv = np.power(rowsum, -1).flatten()#行和倒数,和原文一样的话-2
    r_inv[np.isinf(r_inv)] = 0.#无穷大的等于0
    r_mat_inv = sp.diags(r_inv)#把上面的向量写成对角矩阵D^(-1),和原文一样的话D^(-1/2)
    mx = r_mat_inv.dot(mx)# 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘    #mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def DAD(matrix):
    D = np.diag(np.sum(matrix, axis=1) ** (-0.5))
    D[np.isinf(D)] = 0
    matrix_norm = np.dot(np.dot(D, matrix), D)
    return matrix_norm