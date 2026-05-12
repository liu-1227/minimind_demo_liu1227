import torch
import torch.nn as nn
#nn的方法

'''#随机丢弃，为保持期望不变会改变其他数值
dropout_layer = nn.Dropout(p=0.2)
d1 = torch.Tensor([1, 2, 3, 4, 5])
d2 = dropout_layer(d1)
print(d2)
#示例输出：tensor([0.0000, 2.5000, 3.7500, 0.0000, 6.2500])'''

'''#线性变换：y = wx+b
linear_layer = nn.Linear(in_features=3,out_features=5,bias=True)
l1 = torch.Tensor([1, 2, 3])
l2 = torch.Tensor([4, 5, 6])
output2 = linear_layer(l2)
print(output2)
#输出：tensor([-1.3972,  4.3025, -2.2904,  2.1029,  0.4191], grad_fn=<ViewBackward0>)'''

'''#view:改变形状
v1 = torch.Tensor([[1,2,3,4,5,6],[9,8,7,6,5,4]]) #2,6
v_11 = v1.view(3,4)
print(v_11)
#tensor([[1., 2., 3., 4.],[5., 6., 9., 8.],[7., 6., 5., 4.]])'''


'''#transpose交换维度
t1 = torch.Tensor([[1, 2, 3], [4, 5, 6]])#(2,3)
t11 = t1.transpose(0,1)#(3,2)
print(t11)
#tensor([[1., 4.],[2., 5.],[3., 6.]])'''


'''r1 = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(torch.triu(r1))
#tensor([[1., 2., 3.],[0., 5., 6.],[0., 0., 9.]])
print(torch.triu(r1,diagonal=-1))'''

#reshape
x = torch.arange(1,7)
print(x)#tensor([1, 2, 3, 4, 5, 6])
y = torch.reshape(x,(2,3))
print(y)
#tensor([[1, 2, 3],[4, 5, 6]])

