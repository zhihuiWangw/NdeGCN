import matplotlib.pyplot as plt
import numpy as np
CPS3 = np.array([[1.04, 0.99, 1.11, 0.74, 0.63], [0.25, 0.54, 0.25, 0.49, 0.02]])
EE3 = np.array([[1.01, 1.07, 1.72, 1.03, 1.56], [0.95, 0.74, 0.06, 0.86, 0.78]])
EE4 = np.array([[1.21, 1.62, 1.00, 1.34, 1.6], [0.17, 0.03, 0.98, -0.01, 0.01]])
CPS4 = np.array([[0.15, 0.49, 0.36, 1.02, 0.80], [0.10, 0.06, 0.26, -0.08, -0.01]])
DAWN3 = np.array([[1.35, 1.39, 2.0, 1.36, 1.34], [0.22, 0.18, 0.03, 0.20, 0.23]])
DAWN4 = np.array([[1.04, 1.24, 1.66, 1.65, 2.40], [0.83, 0.21, 0.01, 0.06, 0]])
NDC3 = np.array([[1.51, 2.12, 2.14, 1.03, 1.37], [0.08, 0.04, 0.03, 0.88, 0.19]])
NDC4 = np.array([[2.46, 1.98, 1.65, 2.44, 1.63], [0.02, -0.05, 0.06, 0.02, 0.05]])
CB4 = np.array([[1.30, 1.28, 1.05, 1.09, 1.04], [0.08, 0.13, 0.77, 0.59, 0.82]])
CB3 = np.array([[1.29, 0.61, 0.98, 1.16, 1.11], [0.33, 0.43, 0.90, 0.55, 0.67]])
plt.matshow(CB4, cmap=plt.cm.Greens)#
for i in range(CPS3.shape[0]):
    for j in range(CB4.shape[1]):
        plt.text(x=j, y=i, s=CB4[i, j], fontsize=40, color='black', verticalalignment="center", horizontalalignment="center")
# plt.title('DAWN', fontdict={'weight': 'bold', 'family': 'Times New Roman', 'size': 52}, pad=20) #改变图标题字体
# labels = ['Pos', 'Neg']#contact-primary-school
# plt.yticks([0, 1], labels, fontsize=48, weight='bold', fontproperties='Times New Roman')
plt.xticks([])
plt.yticks([])

plt.show()
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 画第1个图：折线图
# CPS3 = np.array([[1.04, 0.99, 1.11, 0.74, 0.63], [0.25, 0.54, 0.25, 0.49, 0.02]])
# plt.subplot(2,5,1)
# plt.matshow(CPS3, cmap=plt.cm.Blues)
# for i in range(CPS3.shape[0]):
#     for j in range(CPS3.shape[1]):
#         plt.text(x=j, y=i, s=CPS3[i, j], fontsize=48, color='black', verticalalignment="center", horizontalalignment="center")
# plt.title('contact-high-school',fontdict={'weight': 'bold', 'family': 'Times New Roman', 'size': 48}) #改变图标题字体
# labels = ['Pos', 'Neg']
# plt.yticks([0, 1], labels, fontsize=48, weight='bold', fontproperties='Times New Roman')
# plt.xticks([])
#
# # 画第2个图：散点图
# CPS3 = np.array([[1.04, 0.99, 1.11, 0.74, 0.63], [0.25, 0.54, 0.25, 0.49, 0.02]])
# plt.subplot(2,5,2)
# plt.matshow(CPS3, cmap=plt.cm.Blues)
# for i in range(CPS3.shape[0]):
#     for j in range(CPS3.shape[1]):
#         plt.text(x=j, y=i, s=CPS3[i, j], fontsize=48, color='black', verticalalignment="center", horizontalalignment="center")
# plt.title('contact-high-school',fontdict={'weight': 'bold', 'family': 'Times New Roman', 'size': 48}) #改变图标题字体
# labels = ['Pos', 'Neg']
#
# plt.xticks([])
#
# # 画第3个图：饼图
# CPS3 = np.array([[1.04, 0.99, 1.11, 0.74, 0.63], [0.25, 0.54, 0.25, 0.49, 0.02]])
# plt.subplot(2,5,3)
# plt.matshow(CPS3, cmap=plt.cm.Blues)
# for i in range(CPS3.shape[0]):
#     for j in range(CPS3.shape[1]):
#         plt.text(x=j, y=i, s=CPS3[i, j], fontsize=48, color='black', verticalalignment="center", horizontalalignment="center")
# plt.title('contact-high-school',fontdict={'weight': 'bold', 'family': 'Times New Roman', 'size': 48}) #改变图标题字体
# labels = ['Pos', 'Neg']
# plt.yticks([0, 1], labels, fontsize=48, weight='bold', fontproperties='Times New Roman')
# plt.xticks([])
#
# # 画第4个图：条形图
# CPS3 = np.array([[1.04, 0.99, 1.11, 0.74, 0.63], [0.25, 0.54, 0.25, 0.49, 0.02]])
# plt.subplot(2,5,4)
# plt.matshow(CPS3, cmap=plt.cm.Blues)
# for i in range(CPS3.shape[0]):
#     for j in range(CPS3.shape[1]):
#         plt.text(x=j, y=i, s=CPS3[i, j], fontsize=48, color='black', verticalalignment="center", horizontalalignment="center")
# plt.title('contact-high-school',fontdict={'weight': 'bold', 'family': 'Times New Roman', 'size': 48}) #改变图标题字体
# labels = ['Pos', 'Neg']
# plt.yticks([0, 1], labels, fontsize=48, weight='bold', fontproperties='Times New Roman')
# plt.xticks([])
#
# CPS3 = np.array([[1.04, 0.99, 1.11, 0.74, 0.63], [0.25, 0.54, 0.25, 0.49, 0.02]])
# plt.subplot(2,5,5)
# plt.matshow(CPS3, cmap=plt.cm.Blues)
# for i in range(CPS3.shape[0]):
#     for j in range(CPS3.shape[1]):
#         plt.text(x=j, y=i, s=CPS3[i, j], fontsize=48, color='black', verticalalignment="center", horizontalalignment="center")
# plt.title('contact-high-school',fontdict={'weight': 'bold', 'family': 'Times New Roman', 'size': 48}) #改变图标题字体
# labels = ['Pos', 'Neg']
# plt.yticks([0, 1], labels, fontsize=48, weight='bold', fontproperties='Times New Roman')
# plt.xticks([])
#
# CPS3 = np.array([[1.04, 0.99, 1.11, 0.74, 0.63], [0.25, 0.54, 0.25, 0.49, 0.02]])
# plt.subplot(2,5,6)
# plt.matshow(CPS3, cmap=plt.cm.Blues)
# for i in range(CPS3.shape[0]):
#     for j in range(CPS3.shape[1]):
#         plt.text(x=j, y=i, s=CPS3[i, j], fontsize=48, color='black', verticalalignment="center", horizontalalignment="center")
# plt.title('contact-high-school',fontdict={'weight': 'bold', 'family': 'Times New Roman', 'size': 48}) #改变图标题字体
# labels = ['Pos', 'Neg']
# plt.yticks([0, 1], labels, fontsize=48, weight='bold', fontproperties='Times New Roman')
# plt.xticks([])
#
# CPS3 = np.array([[1.04, 0.99, 1.11, 0.74, 0.63], [0.25, 0.54, 0.25, 0.49, 0.02]])
# plt.subplot(2,5,7)
# plt.matshow(CPS3, cmap=plt.cm.Blues)
# for i in range(CPS3.shape[0]):
#     for j in range(CPS3.shape[1]):
#         plt.text(x=j, y=i, s=CPS3[i, j], fontsize=48, color='black', verticalalignment="center", horizontalalignment="center")
# plt.title('contact-high-school',fontdict={'weight': 'bold', 'family': 'Times New Roman', 'size': 48}) #改变图标题字体
# labels = ['Pos', 'Neg']
# plt.yticks([0, 1], labels, fontsize=48, weight='bold', fontproperties='Times New Roman')
# plt.xticks([])
#
# CPS3 = np.array([[1.04, 0.99, 1.11, 0.74, 0.63], [0.25, 0.54, 0.25, 0.49, 0.02]])
# plt.subplot(2,5,8)
# plt.matshow(CPS3, cmap=plt.cm.Blues)
# for i in range(CPS3.shape[0]):
#     for j in range(CPS3.shape[1]):
#         plt.text(x=j, y=i, s=CPS3[i, j], fontsize=48, color='black', verticalalignment="center", horizontalalignment="center")
# plt.title('contact-high-school',fontdict={'weight': 'bold', 'family': 'Times New Roman', 'size': 48}) #改变图标题字体
# labels = ['Pos', 'Neg']
# plt.yticks([0, 1], labels, fontsize=48, weight='bold', fontproperties='Times New Roman')
# plt.xticks([])
#
# CPS3 = np.array([[1.04, 0.99, 1.11, 0.74, 0.63], [0.25, 0.54, 0.25, 0.49, 0.02]])
# plt.subplot(2,5,9)
# plt.matshow(CPS3, cmap=plt.cm.Blues)
# for i in range(CPS3.shape[0]):
#     for j in range(CPS3.shape[1]):
#         plt.text(x=j, y=i, s=CPS3[i, j], fontsize=48, color='black', verticalalignment="center", horizontalalignment="center")
# plt.title('contact-high-school',fontdict={'weight': 'bold', 'family': 'Times New Roman', 'size': 48}) #改变图标题字体
# labels = ['Pos', 'Neg']
# plt.yticks([0, 1], labels, fontsize=48, weight='bold', fontproperties='Times New Roman')
# plt.xticks([])
#
# CPS3 = np.array([[1.04, 0.99, 1.11, 0.74, 0.63], [0.25, 0.54, 0.25, 0.49, 0.02]])
# plt.subplot(2,5,10)
# plt.matshow(CPS3, cmap=plt.cm.Blues)
# for i in range(CPS3.shape[0]):
#     for j in range(CPS3.shape[1]):
#         plt.text(x=j, y=i, s=CPS3[i, j], fontsize=48, color='black', verticalalignment="center", horizontalalignment="center")
# plt.title('contact-high-school',fontdict={'weight': 'bold', 'family': 'Times New Roman', 'size': 48}) #改变图标题字体
# labels = ['Pos', 'Neg']
# plt.yticks([0, 1], labels, fontsize=48, weight='bold', fontproperties='Times New Roman')
# plt.xticks([])
# plt.show()

#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# CPS3 = np.array([[1.04, 0.99, 1.11, 0.74, 0.63], [0.25, 0.54, 0.25, 0.49, 0.02]])
#
# fig, ax = plt.subplots(2, 5)
# # 画第1个图：折线图
# ax[0][0].plt.matshow(CPS3, cmap=plt.cm.Blues)
# for i in range(CPS3.shape[0]):
#     for j in range(CPS3.shape[1]):
#         plt.text(x=j, y=i, s=CPS3[i, j], fontsize=48, color='black', verticalalignment="center", horizontalalignment="center")
# plt.title('contact-high-school',fontdict={'weight': 'bold', 'family': 'Times New Roman', 'size': 48}) #改变图标题字体
# labels = ['Pos', 'Neg']
# plt.yticks([0, 1], labels, fontsize=48, weight='bold', fontproperties='Times New Roman')
# plt.xticks([])
#
# # 画第2个图：散点图
# ax[0][1].scatter(np.arange(0, 10), np.random.rand(10))
# # 画第3个图：饼图
# ax[1][0].pie(x=[15, 30, 45, 10], labels=list('ABCD'), autopct='%.0f', explode=[0, 0.05, 0, 0])
# # 画第4个图：条形图
# ax[1][1].bar([20, 10, 30, 25, 15], [25, 15, 35, 30, 20], color='b')
# plt.show()