import torch

'''#类似三目运算——符合条件的位置是x，不符合条件的位置是y
x = torch.tensor([1,2,4,4,5])
y = torch.tensor([10,20,40,40,50])
condition = x > 3
result = torch.where(condition, x, y)
print(result)
#tensor([10, 20,  4,  4,  5])'''

'''#arange:生成类似等差数列的序列
t1 = torch.arange(0,10,2)
print(t1)
#tensor([0, 2, 4, 6, 8])

t2 = torch.arange(5,1,-1)
print(t2)
#tensor([5, 4, 3, 2])'''

'''#outer:外积（叉乘）
v1 = torch.tensor([1,2,3])
v2 = torch.tensor([4,5,6])
result = torch.outer(v1,v2)
print(result)
#tensor([[ 4,  5,  6],[ 8, 10, 12],[12, 15, 18]])
'''

'''#cat:拼接
c1 = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
#print(c1.shape) #torch.Size([2, 2, 3])
c2 = torch.tensor([[[11,22,33],[44,55,66]],[[77,88,99],[1100,1111,1122]]])
result = torch.cat((c1,c2),dim = 0)
print(result)
#tensor([[[ 1,  2,  3],[ 4,  5,  6]],  [[ 7,  8,  9],[10, 11, 12]], [[ 11,  22,  33],[ 44,  55,  66]], [[ 77,  88,  99],[1100, 1221, 1122]]])
print(result.shape)
#torch.Size([4, 2, 3])

#result = torch.cat((c1,c2),dim = 1)
#print(result)
#tensor([[[ 1,  2,  3],[ 4,  5,  6],[ 11,  22,  33],[ 44,  55,  66]], [[ 7,  8,  9],[10, 11, 12],[ 77,  88,  99],[1100, 1111, 1122]]])
#print(result.shape)
#torch.Size([2, 4, 3])'''


'''#unsqueeze:增加维度
u1 = torch.tensor([1,2,3])
u2 = u1.unsqueeze(0)
print(u2)#tensor([[1, 2, 3]])
print(u1.shape)#torch.Size([3])
print(u2.shape)#torch.Size([1, 3])'''