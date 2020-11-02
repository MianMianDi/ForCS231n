from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# x=torch.empty(5,3)
# x=torch.rand(5,3)
# x=torch.zeros(5,3,dtype=torch.long)
# x=torch.tensor([5.5,3])
# x=x.new_ones(5,3,dtype=torch.double)
# x=torch.randn_like(x,dtype=torch.float)
# # print(x)
# print(x.size()) # return tuple

# y=torch.rand(5,3)
# print(x+y)
# print(torch.add(x,y))
# result=torch.empty(5,3)
# torch.add(x,y,out=result) # return stored in result
# print(result)
# y.add_(x) # add in-place
# print(y)

# x=torch.randn(4,4) # random numbers from a standard normal distribution
# y=x.view(16) # reshape
# z=x.view(-1,8) # the size -1 is inferred from other dimensions
# print(x.size(),y.size(),z.size())

# x=torch.randn(1)
# print(x.item()) # get value if x only has 1 element

# a=torch.ones(5)
# b=a.numpy() # 两者指向的内存为同一个
# print(a)
# print(b)
# a.add_(1)
# print(a)
# print(b)
# a=np.ones(5)
# b=torch.from_numpy(a)
# np.add(a,1,out=a)
# print(a)
# print(b)

# if torch.cuda.is_available():
#     device=torch.device("cuda")
#     y=torch.ones_like(x,device=device)
#     x=x.to(device)
#     z=x+y
#     print('z='%(z))
#     print('z='%(z.to("cpu",torch.double)))

# x=torch.ones(2,2,requires_grad=True)
# print(x)
# y=x+2
# print(y)
# z=y*y*3
# out=z.mean()
# print(z)
# print(out)
# out.backward() # out 为标量 可利用autograd直接求梯度
# print(x.grad) # d(out)/d(o)

# x=torch.randn(3,requires_grad=True)
# y=x*2
# print(y)
# while y.data.norm()<1000:
#     y=y*2
# print(y)
# v=torch.tensor([0.1,1,0.0001],dtype=torch.float)
# y.backward(v)
# print(x.grad)
# with torch.no_grad():# 防止跟踪历史记录
#     print((x**2).requires_grad)

# class Net(nn.Module):
#     def __init__(self):
#         # super(Net,self).__init__()
#         super().__init__() # 和上一句等价，调用超类的构造函数进行子类的初始化
#         # Applies a 2D convolution over an input signal composed of several input planes.
#         # in-channel:1 out-channel:6 kernal-size:5*5
#         self.conv1=nn.Conv2d(1,6,5)
#         self.conv2=nn.Conv2d(6,16,5)
#         # an affine operation: y = Wx + b
#         self.fc1=nn.Linear(16*5*5,120)
#         self.fc2=nn.Linear(120,84)
#         self.fc3=nn.Linear(84,10)

#     def forward(self,x):
#         # 2x2 Max pooling
#         x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
#         # 如果是方阵,则可以只使用一个数字进行定义
#         x=F.max_pool2d(F.relu(self.conv2(x)),2)
#         x=x.view(-1,self.num_flat_features(x))
#         x=F.relu(self.fc1(x))
#         x=F.relu(self.fc2(x))
#         x=self.fc3(x)
#         return x

#     def num_flat_features(self,x):
#         size=x.size()[1:]
#         num_features=1
#         for s in size:
#             num_features*=s
#         return num_features

# net=Net()
# print(net)

# params=list(net.parameters()) # 可学习参数
# print(len(params))
# print(params[0].size())

class MyReLU(torch.autograd.Function):
    """
    我们可以通过建立torch.autograd的子类来实现我们自定义的autograd函数，并完成张量的正向和反向传播。
    """

    @staticmethod
    def forward(ctx, input):
        """
        在前向传播中，我们收到包含输入和返回的张量包含输出的张量。 
        ctx是可以使用的上下文对象存储信息以进行向后计算。 
        您可以使用ctx.save_for_backward方法缓存任意对象，以便反向传播使用。
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        在反向传播中，我们接收到上下文对象和一个张量，其包含了相对于正向传播过程中产生的输出的损失的梯度。
        我们可以从上下文对象中检索缓存的数据，并且必须计算并返回与正向传播的输入相关的损失的梯度。
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

dtype=torch.float
device=torch.device("cpu")

N,D_in,H,D_out=64,1000,100,10

x=torch.randn(N,D_in,device=device,dtype=dtype)
y=torch.randn(N,D_out,device=device,dtype=dtype)

# 使用nn包定义模型和损失函数
model=torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out)
)
loss_fn=torch.nn.MSEloss(reduction='sum')

# 将requires_grad设置为True，意味着我们希望在反向传播时候计算这些值的梯度
w1=torch.randn(D_in,H,device=device,dtype=dtype,requires_grad=True)
w2=torch.randn(H,D_out,device=device,dtype=dtype,requires_grad=True)

learning_rate=1e-6
# 使用optim包定义优化器(Optimizer）。Optimizer将会为我们更新模型的权重
# 这里我们使用Adam优化方法；optim包还包含了许多别的优化算法
# Adam构造函数的第一个参数告诉优化器应该更新哪些张量
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # 前向传播：计算预测值y
    # y_pred=x.mm(w1).clamp(min=0).mm(w2)
    # 我们也使用自定义的自动求导操作来计算 RELU.
    # relu=MyReLU.apply
    # y_pred = relu(x.mm(w1)).mm(w2)
    
    # 模块对象重载了__call__运算符，所以可以像函数那样调用它们
    # 这么做相当于向模块传入了一个张量，然后它返回了一个输出张量
    y_pred = model(x)
    
    # 计算并输出loss
    # loss是一个形状为(1,)的张量
    # loss.item()是这个张量对应的python数值
    # loss=(y_pred-y).pow(2).sum().item()
    
    loss=loss_fn(y_pred,y).item()
    if t%100==99:
        print(t,loss)
    # # 反向传播之前清零梯度
    # model.zero_grad()

    # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零(这些张量是模型可学习的权重)。
    # 这是因为默认情况下，每当调用.backward(）时，渐变都会累积在缓冲区中(即不会被覆盖）
    optimizer.zero_grad()

    # 这个调用将计算loss对所有requires_grad=True的tensor的梯度
    # ,即对所有可学习参数的梯度。
    # 这次调用后，w1.grad和w2.grad将分别是loss对w1和w2的梯度张量。
    loss.backward()

    # 调用Optimizer的step函数使它所有参数更新
    optimizer.step()
    
    # 使用梯度下降更新权重。对于这一步，我们只想对w1和w2的值进行原地改变；不想为更新阶段构建计算图，
    # 所以我们使用torch.no_grad()上下文管理器防止PyTorch为更新构建计算图
    # with torch.no_grad():
    #     w1 -= learning_rate * w1.grad
    #     w2 -= learning_rate * w2.grad

    #     # 反向传播之后手动将梯度置零
    #     w1.grad.zero_()
    #     w2.grad.zero_()

    with torch.no_grad():
        for param in model.parameters():
            param-=learning_rate*param.grad






