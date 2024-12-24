"""
24-ins-Anchor-guided global view reconstruction for multi-view
multi-label feature selection
"""
import time
import numpy as np
from numpy.random import seed
from numpy import linalg as LA

from skfeature.utility.construct_W import construct_W
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler
import pylab as pl
import matplotlib.pyplot as plt

eps = 2.2204e-16

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def view6(X, x_view, Y, dataset, alpha, beta, gamma, lamb,kk):# before setting is 100
    time_start = time.time()
    n_view = len(x_view)
    num, label_num = Y.shape
    _, feature_num = X.shape
    Y_ori = Y.copy()


    x = []
    m = []
    for i in range(n_view):
        m.append(x_view[i][0])
    t1 = 0
    for i in range(0, len(m)):
        if i == 0:
            x.append(X[:, :m[i]])
        else:
            x.append(X[:, t1:(t1 + m[i])]) 
        t1 = t1 + m[i]
    # 初始化
    seed(1)
    s = []
    d = []
    p = []
    k = 20 
    k=kk
    k1 = 10
    B = np.random.rand(num, k)
    W = np.random.rand(feature_num, label_num)
    A1 = np.random.rand(k1, k1)
    A2 = np.random.rand(k1, k1)
    W1 = np.random.rand(num, k1)
    X1 = np.dot(Y, W.T)
    X1 = normalization(X1)

    R1 = np.dot(np.dot(A1,W1.T),B)
    R2 = np.dot(np.dot(A2,W1.T),X1)

    t2 = 0
    for i in range(n_view):
        s1=np.random.rand(m[i], 1)
        ts1=sum(s1)
        for ii in range(len(s1)):
            s1[ii] = s1[ii]/ts1
        ss = np.diag(s1.flat) 
        s.append(ss)
        dd = np.zeros((feature_num,m[i]))
        if i == 0:
            row, col = np.diag_indices(m[i])
            dd[row,col] = np.ones((1,m[i]))
        else:
            row = np.array(list(range(t2,t2+m[i])))
            col = np.array(list(range(m[i])))

            dd[row,col]= np.ones((1,m[i]))
        t2 = t2+m[i]
        d.append(dd)
        pp = np.dot(B.T,x[i])
        p.append(pp)

    Ly_lst = []
    Sy_lst = []
    Ay_lst = []
    options = {'metric': 'euclidean', 'neighbor_mode': 'knn', 'k': 20, 'weight_mode': 'heat_kernel', 't': 1.0}

    Sy = construct_W(Y_ori, **options)#6-9
    Sy = Sy.A
    Ay = np.diag(np.sum(Sy, 0))
    Ly = Ay - Sy
    Sy_lst.append(Sy)
    Ay_lst.append(Ay)
    Ly_lst.append(Ly)

    nu_temp = np.empty(n_view)
    nu_all = 0
    for i in range(n_view):
        nu_temp[i] = (1 / np.trace(x[i].T @ Ly @ x[i]))  
        nu_all = nu_all + nu_temp[i]  
    nu = nu_temp / nu_all  

    cver_lst = []
    obj = []
    obji = 1
    iter = 0

    while 1:
        for i in range(n_view):
            s[i] = np.multiply(s[i], np.true_divide(np.dot(np.dot(x[i].T,X1),d[i]),np.dot(np.dot(x[i].T,x[i]),s[i])+eps))
            s[i] = np.diagonal(s[i]).reshape(m[i], 1)
 
            s[i] = np.diag(s[i].flat)
            p[i] = np.multiply(p[i],np.true_divide(np.dot(B.T,x[i]),np.dot(np.dot(B.T,B),p[i])+eps))

        d_temp = np.sqrt(np.sum(np.multiply(W, W), 1) + eps)
        d1 = 0.5 / d_temp
        E = np.diag(d1.flat)
        W = np.multiply(W,np.true_divide(np.dot(X1.T,Y),np.dot(np.dot(X1.T,X1),W)+lamb*np.dot(E,W)+eps))


        R1 = np.multiply(R1,np.true_divide(np.dot(np.dot(A1.T,W1.T),B),np.dot(np.dot(np.dot(np.dot(A1.T,W1.T),W1),A1),R1)+eps))
        R2 = np.multiply(R2,np.true_divide(np.dot(np.dot(A2.T,W1.T),X1),np.dot(np.dot(np.dot(np.dot(A2.T,W1.T),W1),A2),R2)+eps))
        A1 = np.multiply(A1,np.true_divide(np.dot(np.dot(W1.T,B),R1.T)+A2,np.dot(np.dot(np.dot(np.dot(W1.T,W1),A1),R1),R1.T)+A1+eps))
        A2 = np.multiply(A2,np.true_divide(np.dot(np.dot(W1.T,X1),R2.T)+A1,np.dot(np.dot(np.dot(np.dot(W1.T,W1),A2),R2),R2.T)+A2+eps))
        W1 = np.multiply(W1,np.true_divide(np.dot(np.dot(B,R1.T),A1.T)+np.dot(np.dot(X1,R2.T),A2.T),np.dot(np.dot(np.dot(np.dot(W1,A1),R1),R1.T),A1.T)+np.dot(np.dot(np.dot(np.dot(W1,A2),R2),R2.T),A2.T)+eps))

        q1 = nu[0]*np.dot(x[0],p[0].T)
        q2 = nu[0]*np.dot(np.dot(B,p[0]),p[0].T)
        q3 = nu[0]*np.dot(np.dot(x[0],s[0]),d[0].T)
        q4 = nu[0]*np.dot(np.dot(X1,d[0]),d[0].T)
        for i in range(1,n_view):
            q1 = q1 + nu[i]*np.dot(x[i],p[i].T)
            q2 = q2 + nu[i]*np.dot(np.dot(B,p[i]),p[i].T)
            q3 = q3 + nu[i]*np.dot(np.dot(x[i],s[i]),d[i].T)
            q4 = q4 + nu[i]*np.dot(np.dot(X1,d[i]),d[i].T)


        B = np.multiply(B,np.true_divide(alpha*q1+beta*np.dot(np.dot(W1,A1),R1), alpha*q2+beta*B+eps))
        
        X1 = np.multiply(X1, np.true_divide(np.dot(Y, W.T) + beta * np.dot(np.dot(W1, A2), R2) + gamma*q3,
                                            np.dot(np.dot(X1, W), W.T) + beta * X1 + gamma*q4 +eps))
        
        temp1 = pow(LA.norm(np.dot(X1, W) - Y, 'fro'), 2)

        temp2 = 0
        temp4 = 0
        for i in range(n_view):
            temp2 = temp2 + nu[i] * pow(LA.norm(x[i] - np.dot(B, p[i]), 'fro'), 2)
            temp4 = temp4 + nu[i] * pow(LA.norm(np.dot(X1, d[i]) - np.dot(x[i], s[i]), 'fro'), 2)
            
        temp3 = 0
        temp3 = temp3 + pow(LA.norm(B - np.dot(np.dot(W1, A1), R1), 'fro'), 2)
        temp3 = temp3 + pow(LA.norm(X1 - np.dot(np.dot(W1, A2), R2), 'fro'), 2)
        temp3 = temp3 + pow(LA.norm(A1 - A2, 'fro'), 2)
        temp5 = np.sum(np.sqrt(np.sum(W * W, 1)))


        objectives = temp1 + alpha * temp2 + beta * temp3 + gamma * temp4 + lamb * temp5
        if iter == 0:
            obji = objectives /2
        obj.append(objectives)

        cver = abs((objectives - obji) / float(obji))
        obji = objectives
        cver_lst.append(cver)

        iter = iter + 1
        if (iter > 2 and (cver < 1e-3 or iter == 500)):
            break
    
    
    time_end = time.time()
    running_time = time_end - time_start
    print('the running time of feature selection is {}\n'.format(running_time))
   
    w_2 = LA.norm(W, ord=2, axis=1)
    f_idx = np.argsort(-w_2)

    # 记录参数
    param = dict()
    param['alpha'] = alpha
    param['beta'] = beta
    param['gamma'] = gamma
    param['lamb'] = lamb
    
    record = dict()
    record['method'] = 'view6'
    record['dataset'] = dataset  # 数据集名称
    record['running_time'] = running_time
    record['selected_num'] = None  # 不设定选取的特征数量，由后续的分类阶段选取数量
    record['param'] = param

    record['obj_value'] = np.array(obj).reshape(1, len(obj))
    record['idx'] = f_idx.tolist()  # 序号按照特征的重要性升序排列，最后一个最重要

    
    return record, iter

