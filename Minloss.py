import numpy as np

def minloss_by_random_search(X_train,Y_train,L):
    bestloss=float("inf")
    for num in range(1000):
        W=np.random.randn(10,3073)*0.0001
        loss=L(X_train,Y_train,W)
        if loss<bestloss:
            bestloss=loss
            bestW=W
        print('in attempt %d the loss was %f, best %f'%(num,loss,bestloss))

def minloss_by_random_local_search(X_train,Y_train,L):
    W=np.random.randn(10,3073)*0.001
    bestloss=float("inf")
    step_size=0.0001
    for i in range(1000):
        W_try=W+np.random.randn(10,3073)*step_size
        loss=L(X_train,Y_train,W_try)
        if loss<bestloss:
            bestloss=loss
            W=W_try
        print('iter %d loss is %f'%(i,bestloss))

def eval_numerical_gradient(f,x):
    fx=f(x)
    grad=np.zeros(x.shape)
    h=0.00001

    it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        ix=it.multi_index
        old_value=x[ix]
        x[ix]=old_value+h
        fxh=f(x)
        x[ix]=old_value

        grad[ix]=(fxh-fx)/h
        it.iternext()

    return grad
