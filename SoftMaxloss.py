import numpy as np
def L(f):
    f-=np.max(f)
    p=np.exp(f)/np.sum(np.exp(f))
    loss=-np.log(p)
    N=f.shape[0]
    loss=np.sum(loss)/N
    return loss

