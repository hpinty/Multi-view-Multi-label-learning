
import time
import numpy as np
from numpy.random import seed
from numpy import linalg as LA
from skfeature.utility.construct_W import construct_W
from scipy.spatial.distance import pdist

eps = 2.2204e-16

def kernelmatrix(par, trainX, testX):
    n1sq=np.sum(np.square(testX.T),axis=0)
    t1 = n1sq.shape[0]
    n1sq=n1sq.reshape(1,t1)
    n1 = testX.T.shape[1]
    if ~np.any(trainX):
        print("Y is empty.")
        D = np.dot(np.ones((n1, 1)), n1sq).T + np.dot(np.ones((n1, 1)), n1sq) - 2 * np.dot(testX, testX.T)
    else:
        n2sq=np.sum(np.square(trainX.T),axis=0)
        t2 = n2sq.shape[0]
        n2sq = n2sq.reshape(1, t2)
        n2=trainX.T.shape[1]
        D=np.dot(np.ones((n2,1)),n1sq).T+np.dot(np.ones((n1,1)),n2sq)-2*np.dot(testX,trainX.T)
    H = np.exp(-D/(2*np.square(par)))

    return H

def UGRFS(X, x_view, Y, dataset, alpha, beta, gamma, lamb):
    time_start = time.time()

    # initialization
    n_view = len(x_view)
    num, label_num = Y.shape
    _, feature_num = X.shape
    w = []
    m = []
    c = []

    seed(1)
    for i in range(n_view):
        m.append(x_view[i][0])
        cc = np.diag(np.random.rand(num,1).flat)
        c.append(cc)

    # reconstruction for x-view：
    x = []
    t1 = 0
    for i in range(0, len(m)):
        if i == 0:
            x.append(X[:, :m[i]])
        else:
            x.append(X[:, t1:(t1 + m[i])])
        t1 = t1 + m[i]

    for i in range(n_view):
        w.append(np.random.rand(x[i].shape[1],label_num))

    # graph Laplacian
    Ly_lst = []
    Sy_lst = []
    Ay_lst = []
    options = {'metric': 'euclidean', 'neighbor_mode': 'knn', 'k': 20, 'weight_mode': 'heat_kernel', 't': 1.0}

    Sy = construct_W(Y, **options)
    Sy = Sy.A
    Ay = np.diag(np.sum(Sy, 0))
    Ly = Ay - Sy
    Sy_lst.append(Sy)
    Ay_lst.append(Ay)
    Ly_lst.append(Ly)

    # view weights
    nu_temp = np.empty(n_view)
    nu_all = 0
    for i in range(n_view):
        nu_temp[i] = (1 / np.trace(x[i].T @ Ly @ x[i]))
        nu_all = nu_all + nu_temp[i]
    nu = nu_temp / nu_all

    wx = []
    for i in range(n_view):
        wx.append(x[i] * nu[i])
    new_X = np.concatenate(wx, axis=1)

    par = 1/np.mean(pdist(Y))
    H = kernelmatrix(par,Y,Y)
    UnitMatrix = np.ones((num,1))
    Yx = np.concatenate([H,UnitMatrix],axis=1)

    wy = []
    for i in range(n_view):
        wy.append(np.random.rand(Yx.shape[1], x[i].shape[1]))


    cver_lst = []
    obj = []
    obji = 1
    iter = 0

    while 1:

        for i in range(n_view):

            d_temp = np.sqrt(np.sum(np.multiply(w[i], w[i]), 1) + eps)
            d1 = 0.5 / d_temp
            D = np.diag(d1.flat)

            p1 = np.dot(c[i],x[i])
            w[i] = np.multiply(w[i], np.true_divide(np.dot(p1.T,Y), np.dot(np.dot
            (p1.T, p1), w[i]) + lamb * np.dot(D, w[i]) + eps))

            c[i] = np.multiply(c[i], np.true_divide(beta*np.dot(np.dot(Yx,wy[i]),x[i].T)+np.dot(np.dot(Y,w[i].T),x[i].T),beta*np.dot(np.dot(c[i],x[i]),x[i].T)+np.dot(np.dot(np.dot(p1,w[i]),w[i].T),x[i].T)+eps))
            c[i] = np.diagonal(c[i]).reshape(num, 1)

            c[i] = np.diag(c[i].flat)
            # updating Wy_i：
            p3 = x[i] * nu[i]
            wy[i] = np.multiply(wy[i],np.true_divide(gamma*np.dot(Yx.T,p3)+alpha*np.dot(np.dot(np.dot(Yx.T,Sy),Yx),wy[i])+beta*np.dot(np.dot(Yx.T,c[i]),x[i]),alpha*np.dot(np.dot(np.dot(Yx.T,Ay),Yx),wy[i])+(gamma+beta)*np.dot(np.dot(Yx.T,Yx),wy[i])+eps))


        # objective function:
        Wy = np.concatenate(wy, axis=1)
        temp1 = pow(LA.norm(np.dot(Yx, Wy) - new_X, 'fro'), 2)
        p2 = np.dot(Yx, Wy)
        temp2 = np.trace(np.dot(np.dot(p2.T, Ly), p2))
        temp3 = 0
        temp4 = 0

        for i in range(n_view):
            p4 = np.dot(c[i], x[i])
            temp3 = temp3 + pow(LA.norm(np.dot(Yx, wy[i]) - p4, 'fro'), 2)
            temp4 = temp4 + pow(LA.norm(np.dot(p4, w[i]) - Y, 'fro'), 2)

        W = np.concatenate(w, axis=0)
        temp5 = np.sum(np.sqrt(np.sum(W * W, 1)))
        objectives = temp4 + alpha * temp2 + beta * temp3 + gamma * temp1 + lamb * temp5

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

    w_2 = LA.norm(W, ord=2, axis=1)
    f_idx = np.argsort(-w_2)

    param = dict()
    param['alpha'] = alpha
    param['beta'] = beta
    param['gamma'] = gamma
    param['lamb'] = lamb
    # return the info. (paras, ranked no. of features and running time)
    record = dict()
    record['method'] = 'UGRFS'
    record['dataset'] = dataset
    record['running_time'] = running_time
    record['selected_num'] = None
    record['param'] = param
    record['obj_value'] = np.array(obj).reshape(1, len(obj))
    record['idx'] = f_idx.tolist()

    return record, iter