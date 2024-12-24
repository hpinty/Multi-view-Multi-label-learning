import time
import numpy as np
from numpy.random import seed
from numpy import linalg as LA
from skfeature.utility.construct_W import construct_W
from sklearn.preprocessing import MinMaxScaler
import pylab as pl

eps = 2.2204e-16


def I2VSLC(X, x_view, Y, dataset, alpha, beta, gamma, lamb):
    time_start = time.time()
    n_view = len(x_view)
    num, label_num = Y.shape
    w = []  # 参数矩阵W
    m = []
    y = []

    seed(100)
    for i in range(n_view):
        m.append(x_view[i][0])
        w.append(np.random.rand(m[i], label_num))
        y.append(np.random.randint(2, size=(num, label_num)))

    AA = np.diag([1]*label_num)

    # 切分：
    v = []
    t1 = 0
    for i in range(0, len(m)):
        if i == 0:
            v.append(X[:, :m[i]])
        else:
            v.append(X[:, t1:(t1 + m[i])])
        t1 = t1 + m[i]

    # 不同视图结构相似性
    Lx_lst = []
    Sx_lst = []
    Ax_lst = []
    options = {'metric': 'euclidean', 'neighbor_mode': 'knn', 'k': 5, 'weight_mode': 'heat_kernel', 't': 1.0}
    for i in range(0, len(m)):
        Sx = construct_W(v[i], **options)
        Sx = Sx.A
        Ax = np.diag(np.sum(Sx, 0))
        Lx = Ax - Sx
        Sx_lst.append(Sx)
        Ax_lst.append(Ax)
        Lx_lst.append(Lx)

    obj = []
    obji = 1
    iter = 0
    cver_lst = []

    sum_Y = y[0].copy()
    for i in range(1, n_view):
        sum_Y = sum_Y + y[i]
    sum_Yt = sum_Y.copy()
    sum_Yt[sum_Yt == 0] = 0
    sum_Yt[sum_Yt > 0] = 1

    while 1:

        for i in range(n_view):
            d_temp = np.full((m[i], 1), LA.norm(w[i], 'fro'))
            d2 = 0.5 / d_temp
            D = np.diag(d2.flat)

            c_temp = np.sqrt(np.sum(np.multiply(w[i], w[i]), 1) + eps)
            c1 = 0.5 / c_temp
            C = np.diag(c1.flat)

            w[i] = np.multiply(w[i], np.true_divide(np.dot(v[i].T, Y), np.dot
            (v[i].T, np.dot(v[i], w[i])) + gamma*np.dot(D, w[i]) + lamb*np.dot(C, w[i]) + eps))


        # update A
        AA = np.multiply(AA, np.true_divide(np.dot(Y.T, sum_Yt) + np.dot(Y.T, Y),
                        2 * np.dot(np.dot(Y.T, Y), AA) + eps))


        for i in range(n_view):
            tem1 = sum_Yt - y[i]
            tem1[tem1 == 0] = 0 
            tem1[tem1 > 0] = 1  
            tem1[tem1 < 0] = 1
            tem2 = y[i] - sum_Yt
            tem2[tem2 == 0] = 0  
            tem2[tem2 > 0] = 1  
            tem2[tem2 < 0] = 1
            y[i] = np.multiply(y[i], np.true_divide(
                alpha*np.dot(Sx_lst[i], y[i]) + alpha*(n_view-1)*(tem1) + np.dot(v[i], w[i]) + beta*(np.dot(Y,AA)+ (tem2)),
                alpha*np.dot(Ax_lst[i], y[i]) + alpha*(n_view-1) * y[i] + y[i]+ beta*y[i] + eps))
        for i in range(n_view):

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(y[i])  
            y[i] = scaler.transform(y[i])

            y[i][y[i] <= 0.5] = 0
            y[i][y[i] > 0.5] = 1

        # objectives:
        temp1 = 0
        temp2 = 0
        temp4 = 0

        for i in range(n_view):
            temp1 = temp1 + pow(LA.norm(np.dot(v[i], w[i]) - y[i], 'fro'), 2)
        for i in range(n_view):
            temp2 = temp2 + np.trace(np.dot(np.dot(y[i].T, Lx_lst[i]), y[i]))

            for j in range(i+1, n_view):
                temp2 = temp2 + pow(LA.norm(y[i] - y[j], 'fro'), 2)
        if np.isnan(temp2):
            print("temp2-nan")
            break


        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.dot(Y,AA))  
        t1 = scaler.transform(np.dot(Y,AA))
        t1[t1 <= 0.5] = 0
        t1[t1 > 0.5] = 1


        sum_Y = y[0].copy()
        for i in range(1, n_view):
            sum_Y = sum_Y + y[i]
        sum_Yt = sum_Y.copy()
        sum_Yt[sum_Yt == 0] = 0
        sum_Yt[sum_Yt > 0] = 1

        tem31 = t1 - sum_Yt
        tem31[tem31 == 0] = 0
        tem31[tem31 > 0] = 1
        tem31[tem31 < 0] = 1
        temp3 = pow(LA.norm(tem31, 'fro'), 2)


        tem32 = Y-t1
        tem32[tem32 == 0] = 0 
        tem32[tem32 > 0] = 1  
        tem32[tem32 < 0] = 1

        temp3 += pow(LA.norm(tem32, 'fro'),2)

        for i in range(n_view):
            temp4 = temp4 + LA.norm(w[i], 'fro')

        # update B
        B = np.concatenate(w, axis=0)
        temp5 = np.sum(np.sqrt(np.sum(B * B, 1)))

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

    B = np.concatenate(w, axis=0)
    w_2 = LA.norm(B, ord=2, axis=1)
    f_idx = np.argsort(-w_2)

    # 记录参数
    param = dict()
    param['alpha'] = alpha
    param['beta'] = beta
    param['gamma'] = gamma
    param['lamb'] = lamb

    record = dict()
    record['method'] = 'I2VSLC'
    record['dataset'] = dataset  # 数据集名称
    record['param'] = param
    record['running_time'] = running_time
    record['obj_value'] = np.array(obj).reshape(1, len(obj))
    record['idx'] = f_idx.tolist()  # 序号按照特征的重要性升序排列，最后一个最重要
    record['selected_num'] = None  # 不设定选取的特征数量，由后续的分类阶段选取数量

    return record, iter
