"""
main function from MM25-Tensor-based Opposing yet Complementary learning for Multi-view Multi-label Feature Selection.
input: matrices and hyper-paras is must.
output: the no. of selected features.
"""
import time
import numpy as np
from numpy.random import seed
from numpy import linalg as LA
from scipy.fftpack import fft
from scipy.linalg import svd
from sklearn.preprocessing import MinMaxScaler

eps = 2.2204e-16

def prox_weight_tensor_nuclear_norm(Y,C):
    n1, n2, n3 = Y.shape
    X = np.zeros((n1, n2, n3), dtype=complex)
    Y = fft(Y, axis=2)
    wtnn = 0
    trank = 0

    # 第一个前切片
    U, S, Vh = svd(Y[:, :, 1], full_matrices=False)
    V = Vh.conj().T
    temp = (S - eps) ** 2 - 4 * (C - eps * S)
    ind = np.where(temp > 0)[0]
    r = len(ind)
    if r >= 1:
        S = np.maximum(S[ind] - eps + np.sqrt(temp[ind]), 0) / 2
        X[:, :, 1] = U[:, :r] @ np.diag(S) @ V[:, :r].T
        wtnn += np.sum(S * (C / (S + eps)))
        trank = max(trank, r)

    # i=2,...,halfn3
    halfn3 = round(n3 / 2)
    for i in range(1, halfn3):
        U, S, Vh = svd(Y[:, :, i], full_matrices=False)
        V = Vh.conj().T
        temp = (S - eps) ** 2 - 4 * (C - eps * S)
        ind = np.where(temp > 0)[0]
        r = len(ind)
        if r >= 1:
            S = np.maximum(S[ind] - eps + np.sqrt(temp[ind]), 0) / 2
            X[:, :, i] = U[:, :r] @ np.diag(S) @ V[:, :r].T
            wtnn += np.sum(S * (C / (S + eps)))
            trank = max(trank, r)
        X[:, :, n3 - i] = np.conj(X[:, :, i])

    # 如果 n3 是偶数
    if n3 % 2 == 0:
        i = halfn3 + 1
        U, S, Vh = svd(Y[:, :, i], full_matrices=False)
        V = Vh.conj().T
        temp = (S - eps) ** 2 - 4 * (C - eps * S)
        ind = np.where(temp > 0)[0]
        r = len(ind)
        if r >= 1:
            S = np.maximum(S[ind] - eps + np.sqrt(temp[ind]), 0) / 2
            X[:, :, i] = U[:, :r] @ np.diag(S) @ V[:, :r].T
            wtnn += np.sum(S * (C / (S + eps)))
            trank = max(trank, r)

    newX = np.fft.ifftn(X)
    wtnn /= n3

    return np.real(newX),wtnn,trank

def view7(X, x_view, Y, dataset, alpha, beta, gamma, lamb):
    seed(2)

    time_start = time.time()
    n_view = len(x_view)
    num, label_num = Y.shape
    _, feature_num = X.shape
    aaa = 1
    delta = 1
    rho = 1

    # SPLITTING：
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

    # Initialization
    y = []
    d = []
    t2 = 0
    for i in range(n_view):
        y.append(np.random.randint(2, size=(num, label_num)))
        dd = np.zeros((m[i],feature_num))
        if i == 0:
            row, col = np.diag_indices(m[i])
            dd[row, col] = np.ones((1, m[i]))
        else:
            # """
            col = np.array(list(range(t2, t2 + m[i])))
            row = np.array(list(range(m[i])))

            dd[row, col] = np.ones((1, m[i]))

        t2 = t2 + m[i]
        d.append(dd)


    W = np.random.rand(feature_num,label_num)

    sum_y = 0
    for i in range(n_view):
        sum_y = sum_y + y[i]

    sum_yt = sum_y.copy()
    sum_yt[sum_yt == 0] = 0
    sum_yt[sum_yt > 0] = 1

    Y_n = np.random.randint(2, size=(num, label_num))
    P = np.random.rand(num, num)
    Z = np.zeros((num, num, n_view))
    HH = np.zeros((num, num, n_view))
    kk = int(0.8*label_num)
    U = np.random.rand(kk, label_num)
    V = np.random.rand(feature_num, kk)

    # START
    cver_lst = []
    obj = []
    obji = 1
    iter = 0

    I = np.ones((num,kk))
    while 1:
        # 1.1 Updating y
        for i in range(n_view):
            y_o = sum_yt - y[i]
            y_o[y_o == 0] = 0
            y_o[y_o > 0] = 1

            F = Y - Y_n - y_o
            F[F == 0] = 0
            F[F > 0] = 1
            F[F < 0] = 1

            y[i] = np.multiply(y[i], np.true_divide(delta * np.dot(P, np.dot(x[i], np.dot(d[i], W))) + aaa * np.dot(Z[:, :, i], y[i])  + gamma * F,delta * y[i] + gamma * y[i]  + aaa * np.dot(
                                       np.dot(y[i], y[i].T), y[i]) + eps))


        for i in range(n_view):
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(y[i])
            y[i] = scaler.transform(y[i])
            y[i][y[i] <= 0.5] = 0
            y[i][y[i] > 0.5] = 1

            HH[:, :, i] = np.dot(y[i], y[i].T)

        sum_y = 0
        for i in range(n_view):
            sum_y = sum_y + y[i]
        sum_yt = sum_y.copy()
        sum_yt[sum_yt == 0] = 0
        sum_yt[sum_yt > 0] = 1

        # 1.2 Updating T
        HH2 = HH.transpose((0, 2, 1))
        Z2, wtnn, trank = prox_weight_tensor_nuclear_norm(HH2, rho)
        Z = Z2.transpose((0, 2, 1))

        # 1.3 Updating W
        NL = 1 / (I + np.exp(-np.dot(X, V)))
        d_temp = np.sqrt(np.sum(np.multiply(W, W), 1) + eps)
        d2 = 0.5 / d_temp
        D2 = np.diag(d2.flat)
        tem1 = 0
        tem2 = 0
        for i in range(n_view):
            tem1 = tem1 + np.dot(np.dot(d[i].T, np.dot(x[i].T, P.T)), y[i])
            tem2 = tem2 + np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(d[i].T,x[i].T),P.T),P),x[i]),d[i]),W)
        W = np.multiply(W, np.true_divide(delta*tem1+alpha*np.dot(np.dot(np.dot(X.T,P.T),NL),U),
                                                    delta*tem2+ lamb * np.dot(D2, W) + alpha*np.dot(np.dot(np.dot(np.dot(X.T,P.T),P),X),W)+eps))    

        # 1.4 Updating P
        tem1 = 0
        tem2 = 0
        for i in range(n_view):
            tem1 = tem1 + np.dot(np.dot(y[i], np.dot(d[i],W).T), x[i].T)
            tem2 = tem2 + np.dot(np.dot(np.dot(np.dot(P, x[i]), np.dot(d[i],W)), np.dot(d[i],W).T), x[i].T)

        A = np.dot(P,X)
        P = np.multiply(P, np.true_divide(delta * tem1 + alpha * np.dot(np.dot(np.dot(NL, U),W.T),X.T),delta * tem2 + beta * P + alpha *np.dot(np.dot(np.dot(A, W),W.T),X.T) + eps))

        # 1.5 Updating Y_n
        tem3 = Y - sum_yt
        tem3[tem3 == 0] = 0
        tem3[tem3 > 0] = 1
        tem3[tem3 < 0] = 1
        Y_n = np.multiply(Y_n, np.true_divide(tem3, 2 * Y_n + eps))

        # 1.6 Updating U
        NL = 1 / (I + np.exp(-np.dot(X, V)))
        U = np.multiply(U, np.true_divide(np.dot(NL.T,np.dot(np.dot(P,X),W)),np.dot(np.dot(NL.T,NL),U)+2*U+eps))

        # 1.7 Updating V
        Q = np.exp(-np.dot(X, V)) / ((I + np.exp(-np.dot(X, V))) * (I + np.exp(-np.dot(X, V))))
        V =np.multiply(V, np.true_divide(np.dot(X.T,Q *np.dot(np.dot(P,np.dot(X, W)), U.T)),
                                         np.dot(X.T,Q *np.dot(np.dot(NL, U),U.T))+2*V+eps))


        # 2-Objective function：
        temp1 = 0
        temp2 = 0
        for i in range(n_view):
            temp1 = temp1 + pow(LA.norm(np.dot(P, np.dot(x[i], np.dot(d[i],W))) - y[i], ord='fro'), 2)
            temp2 = temp2 + pow(LA.norm(Z[:, :, i] - HH[:, :, i], ord='fro'), 2)
        temp3 = 0.5*temp2+wtnn

        NL = 1 / (I + np.exp(-np.dot(X,V)))
        tem4 = np.dot(np.dot(P,X),W)-np.dot(NL,U)
        temp4 = pow(LA.norm(tem4,ord='fro'),2)
        temp5 = pow(LA.norm(U, ord='fro'), 2)+pow(LA.norm(V, ord='fro'), 2)
        temp6=pow(LA.norm(P,ord='fro'),2)

        tem7 = sum_yt + Y_n - Y
        tem7[tem7 == 0] = 0
        tem7[tem7 > 0] = 1
        tem7[tem7 < 0] = 1
        temp7 = pow(LA.norm(tem7, ord='fro'), 2)
        temp8 = pow(LA.norm(Y_n, ord='fro'), 2)

        d_temp = np.sqrt(np.sum(np.multiply(W, W), 1) + eps)
        d3 = 0.5 / d_temp
        D3 = np.diag(d3.flat)
        temp9 = np.trace(np.dot(np.dot(W.T, D3), W))

        objectives = delta*temp1 + aaa*(temp3) + alpha * (temp4 + temp5) + \
                     2 * beta * temp6 + gamma * (temp7 + temp8) + 2 * lamb * temp9



        if iter == 0:
            obji = objectives / 2
        obj.append(objectives)

        cver = abs((objectives - obji) / float(obji))
        obji = objectives
        cver_lst.append(cver)

        iter = iter + 1
        if (iter > 2 and (cver < 1e-6 or iter == 500)):
            break

    time_end = time.time()
    running_time = time_end - time_start

    w_2 = LA.norm(W, ord=2, axis=1)
    f_idx = np.argsort(-w_2)

    # Record
    param = dict()
    param['alpha'] = alpha
    param['beta'] = beta
    param['gamma'] = gamma
    param['lamb'] = lamb

    record = dict()
    record['method'] = 'TOCL'
    record['dataset'] = dataset
    record['running_time'] = running_time
    record['selected_num'] = None
    record['param'] = param

    record['obj_value'] = np.array(obj).reshape(1, len(obj))
    record['idx'] = f_idx.tolist()

    return record, iter

