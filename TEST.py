import torch

x = torch.ones((112121, 1, 40, 40))  # 第二个是in_channel
print(x)
print(x.size())
print(x.shape)