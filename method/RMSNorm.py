import torch

#开方求导数
t1 = torch.rsqrt(torch.tensor(4.0))
print(t1)

#创建一个全1张量
t2 = torch.one(3,4)
print(t2)