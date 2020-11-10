import numpy as np

from math import sqrt

# nums = {int(sqrt(x)) for x in range(30)}
# print(nums)

# a = np.array([1, 2, 3]) # column vector
# print(type(a))
# print(a.shape)
# print(a[0], a[1], a[2])
# b = np.array([[1, 2, 3], [4, 5, 6]])
# print(b.shape)

# a=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
# row_r1=a[1,:]
# row_r2=a[1:2,:]
# print(row_r1,row_r1.shape)
# print(row_r2,row_r2.shape)

# a=np.array([[1,2],[3,4],[5,6]])
# print(a.shape)
# print(a)
# print(a[[0,1,2],[0,1,0]])
# print(np.array([a[0,0],a[1,1],a[2,0]]))

# x=np.array([[1,2],[3,4]])
# y=np.array([[5,6],[7,8]])
# print(x.dot(y))
# print(np.dot(x,y)) # matrix multiply
# print(np.sum(x,axis=1))
# print(x*y) # elementwise product
# print(np.multiply(x,y))
# print(x.T)

# x=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
# v=np.array([1,0,1])
# y=np.empty_like(x)
# for i in range(4):
#     y[i,:]=x[i,:]+v

# print(y)
# yy=np.tile(v,(4,1))
# y=x+yy
# print(y)

# v=np.array([1,2,3])
# w=np.array([4,5])
# print(v)
# print(v.shape) # 1-dimension
# print(np.reshape(v,(3,1)))
# print(np.reshape(v,(3,1)).shape) # 2-dimension

# from imageio import imread,imsave
# from skimage.transform import resize
# img=imread('cat.jpg')
# print(img.dtype,img.shape)
# img_tinted=img*[1,0.95,0.9]
# img_tinted=resize(img_tinted,(300,300))
# imsave('cat_tinted.jpg',img_tinted)

# x=np.array([[1,2,3],[4,5,6]])
# b=[1,2,3]
# y=np.array([1,2,3])
# d=np.sum(np.abs(x-y),axis=1)
# print(d)
# min_index=np.argmin(d)
# print(min_index)
# mat=np.argmax(x,axis=1)
# mat=x+b
D=3
H=2
mat=np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
print(mat)

# x=np.array([[1,2,3],[4,5,6]])
# print(np.array([1,0,1]))
# z=np.array([0,1,2])
# y=np.array([[0,1,1],z])
# print(y)
# print(x)
# # print(x[y])
# print(x[y[0],y[1]])
# print(x-x[y[0],y[1]]+1)

# it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
# while not it.finished:
#         ix=it.multi_index
#         old_value=x[ix]
#         # z[ix]=old_value+h
#         print(ix)
#         print(x[ix])
#         # fxh=f(x)
#         # z[ix]=old_value

#         # grad[ix]=(fxh-fx)/h
#         it.iternext()    


# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# for y,cls in enumerate(classes):
#     print(y,cls)

# X=np.zeros((5,1))
# X[1]=1
# print(X)
# Y=np.ones((5,3))
# print(Y-X)

