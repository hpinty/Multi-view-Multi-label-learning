import time
import numpy as np
from numpy.random import seed
from numpy import linalg as LA
from skfeature.utility.construct_W import construct_W
from sklearn.preprocessing import MinMaxScaler
import pylab as pl
eps = 2.2204e-16


def EF2FS(X, x_view, Y, dataset, alpha, beta, gamma, lamb, V_dim):
    time_start = time.time()
    n_view = len(x_view)
    num, dim = X.shape
    num, label_num = Y.shape
    w = []  # 参数矩阵W
    m = []
    v = []

    seed(4)
    for i in range(n_view):
        m.append(x_view[i][0])
        w.append(np.random.rand(m[i], V_dim))
    t1 = 0
    for i in range(0, len(m)):
        if i == 0:
            v.append(X[:, :m[i]])
        else:
            v.append(X[:, t1:(t1 + m[i])])
        t1 = t1 + m[i]

    V = np.random.rand(num, V_dim)
    B = np.random.rand(label_num, V_dim)
    A = np.random.rand(dim, V_dim)
 
    cver_lst = []
    # 初始化
    obj = []
    obji = 1
    iter = 0
    while 1:
        # 视图权重系数
        nu_temp=np.empty(n_view)
        nu_all = 0
        for i in range(n_view):
            nu_temp[i] = (1 / np.trace(V.T @ Lx_lst[i] @ V))  
            nu_all = nu_all + nu_temp[i]  
        nu = nu_temp / nu_all  

        wx = []
        for i in range(n_view):
            wx.append(v[i] * nu[i])
        new_X = np.concatenate(wx, axis=1)


        B = np.multiply(B, np.true_divide(np.dot(Y.T, V), np.dot(np.dot(B, V.T), V)+eps))
        xw = np.dot(v[0],w[0])
        for i in range(1,n_view):
            xw = xw + np.dot(v[i],w[i])
        V = np.multiply(V, np.true_divide(alpha*np.dot(Y, B) + beta*xw + np.dot(new_X, A), alpha*np.dot(np.dot(V,B.T),B)+(n_view*beta+1)*V +eps))

        W = np.concatenate(w, axis=0)
       
        Btmp = np.sqrt(np.sum(np.multiply(A, A), 1) + eps)
        d1 = 0.5 / Btmp
        D = np.diag(d1.flat)
        A = np.multiply(A, np.true_divide(np.dot(X.T, V) + gamma * W, np.dot(np.dot(X.T, X), A) + gamma * A +lamb*np.dot(D, A)+ eps))  # 0810
        
        new_a = []
        for i in range(0, len(m)):
            if i == 0:
                t1 = A[:m[i], :]
                new_a.append(t1)
            else:
                t2 = A[m[i - 1]:(m[i - 1] + m[i]),:]
                new_a.append(t2)


        for i in range(n_view):
            w[i]= np.multiply(w[i], np.true_divide(beta*np.dot(v[i].T,V)+gamma*new_a[i],beta*np.dot(np.dot(v[i].T,v[i]),w[i])+gamma*w[i]+eps))#0531

       # 2-目标函数：
        temp1 = pow(LA.norm(np.dot(new_X,A)-V, 'fro'), 2)
        temp2 = pow(LA.norm(Y - np.dot(V,B.T), 'fro'), 2)
        temp3 = 0
        for i in range(n_view):
            temp3 = temp3 + pow(LA.norm(np.dot(v[i],w[i])-V, 'fro'), 2)
        temp4 = 0
        
        new_w = []
        for i in range(0, len(m)):
            if i == 0:
                new_w.append(A[:m[i],:])
            else:
                new_w.append(A[m[i - 1]:(m[i - 1] + m[i]),:])

        for i in range(n_view):
            temp4 = temp4 + pow(LA.norm(new_w[i]-w[i], 'fro'), 2)
       
        temp5 = np.sum(np.sqrt(np.sum(A * A, 1)))
       
        objectives = temp1 + alpha * temp2 + beta * temp3 + gamma * temp4 + lamb * temp5
        if iter == 0:
            obji = objectives /2

        obj.append(objectives)
        cver = abs((objectives - obji) / float(obji))

        obji = objectives
        cver_lst.append(cver)

        iter = iter + 1
        if (iter > 2 and (cver < 1e-3 or iter == 100)):
            break

    time_end = time.time()
    running_time = time_end - time_start
   
    w_2 = LA.norm(A, ord=2, axis=1)
    f_idx = np.argsort(-w_2)

    # 记录参数
    param = dict()
    param['alpha'] = alpha
    param['beta'] = beta
    param['gamma'] = gamma
    param['lamb'] = lamb

    record = dict()
    record['method'] = 'EF2FS'
    record['dataset'] = dataset  # 数据集名称
    record['param'] = param
    record['running_time'] = running_time
    record['obj_value'] = np.array(obj).reshape(1, len(obj))
    record['idx'] = f_idx.tolist()  # 序号按照特征的重要性升序排列，最后一个最重要
    record['selected_num'] = None  # 不设定选取的特征数量，由后续的分类阶段选取数量

    return record, iter