import torch
import numpy as np
import torch.nn as nn
import math
import torch.tensor as tensor

m = nn.Sigmoid()

loss = nn.BCELoss(reduction='none')

input = tensor(np.array([[0.6682],[0.6042],[0.7042]],dtype=np.float32))
target = tensor(np.array([[0],[1],[1]],dtype=np.float32))
lossinput = m(input)
output = torch.mean(loss(lossinput, target))

print("输入值:")
print(lossinput)
print("输出的目标值:")
print(target)
print("计算loss的结果:")
print(output)
print("自己计算的第一个loss：")
print(-(target[0]*math.log(lossinput[0])+(1-target[0])*math.log(1-lossinput[0])))
print(torch.mean(torch.ones(5,1)))