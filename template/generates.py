import numpy as np
import matplotlib as plt
seed = 2
def generateds():
    #基于seed产生随机数
    rdm = np.random.seed(seed)
    X = rdm.randn(300,2)    
    Y_ = [int (x0*x0+x1*x1<2) for (x0,x1) in X]
    Y_c=[['red ' if y else 'blue'] for y in Y_]
    X=np.vstack(X).reshape(-1,2)
    Y_=np.vstack(Y_).reshape(-1,2)
    return X , Y_, Y_c