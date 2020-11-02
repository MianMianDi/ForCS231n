from __future__ import print_function
import torch
import numpy as np

# x=torch.empty(5,3)
x=torch.rand(5,3)
x=torch.zeros(5,3,dtype=torch.long)
x=torch.tensor([5.5,3])
x=x.new_ones(5,3,dtype=torch.double)
x=torch.randn_like(x,dtype=torch.float)
# print(x)
print(x.size()) # return tuple

y=torch.rand(5,3)
print(x+y)
print(torch.add(x,y))
result=torch.empty(5,3)
torch.add(x,y,out=result) # return stored in result
print(result)
y.add_(x) # add in-place
print(y)

x=torch.randn(4,4) # random numbers from a standard normal distribution
y=x.view(16) # reshape
z=x.view(-1,8) # the size -1 is inferred from other dimensions
print(x.size(),y.size(),z.size())

x=torch.randn(1)
print(x.item()) # get value if x only has 1 element

a=torch.ones(5)
b=a.numpy() # 两者指向的内存为同一个
print(a)
print(b)
a.add_(1)
print(a)
print(b)
a=np.ones(5)
b=torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)

if torch.cuda.is_available():
    device=torch.device("cuda")
    y=torch.ones_like(x,device=device)
    x=x.to(device)
    z=x+y
    print(z)
    print(z.to("cpu",torch.double))











