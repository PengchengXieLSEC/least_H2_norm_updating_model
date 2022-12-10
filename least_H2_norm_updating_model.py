"""
Description:
Codes for constructing least H^2 norm updating quadratic model 
Authors: Pengcheng Xie, Ya-xiang Yuan Connect: xpc@lsec.cc.ac.cn
"""
import numpy as np
from functools import reduce
import math
from scipy import linalg as LA


def _obtain_coeffs(W, tol_svd, b, option):
   
    if option == 'partial':
        U, S, VT = LA.svd(W)
    else:
        U, S, VT = LA.svd(W, full_matrices=False)

    # Check the condition number
    indices = S < tol_svd
    S[indices] = tol_svd
    Sinv = np.diag(1 / S)
    V = VT.T
    # Coefficients
    lambdacg = reduce(np.dot, [V, Sinv, U.T, b])
    return (lambdacg)


def quad_model_Htwo(X, F_values, args):
    """
    Construct a quadratic model
    """
    
    eps = np.finfo(float).eps
    tol_svd = eps ** 6
    
    (n, m) = X.shape

    G = np.zeros((n, n))
    g = np.zeros((n, 1))

    # Shift the points
    Y = X - np.dot(np.diag(X[:, 0]), np.ones((n, m)))

    if (m < (n + 1) * (n + 2) / 2):
        c1 = args.c1
        c2 = args.c2
        c3 = args.c3
        r  = args.r

        omega1 = (c1 * r ** 4 / (2 * (n + 4) * (n + 2)) + c2 * r ** 2 / (n + 2) + c3)
        omega2 = (c1 * r ** 2 / (n + 2) + c2)
        omega3 = c1 * r ** 4 / (4 * (n + 4) * (n + 2))
        omega4 = c1 * r ** 2 / (n + 2)
        omega5 = c1
        
        b = np.vstack((F_values, np.zeros((n + 1, 1))))
       
        A = 1 / (8 * omega1) * (np.dot(Y.T, Y) ** 2)
        for i in range(len(A)):
            for j in range(len(A)):
                A[i, j] = A[i, j] - (omega3 / (8 * omega1 * (n * omega3 + omega1))) * np.dot(Y[:, i].T,Y[:, i]) * np.dot(Y[:, j].T, Y[:, j])

        J = np.zeros((m, 1))
        
        for i in range(m):
            J[i] = 1 - omega4 / (4 * omega1 + 4 * n * omega3) * np.dot(Y[:, i].T, Y[:, i])
        

        line1 = np.hstack((A, J, Y.T))  
        line2 = np.hstack((J.T, ((n * omega4 ** 2) / (2 * n * omega3 + 2 * omega1) - 2 * omega5) * np.ones((1, 1)), np.zeros((n, 1)).T))
        line3 = np.hstack((Y, np.zeros((n, 1)), -2 * omega2 * np.identity(n)))

        W = np.vstack((line1, line2, line3))

        lambdacg = _obtain_coeffs(W, tol_svd, b, option='partial')  

        # Grab the coeffs of linear terms and the ones of quadratic terms
        
        c = lambdacg[m:m + 1]
        g = lambdacg[m + 1:m + 1 + n]
        

        G = np.zeros((n, n))  
        
        inner_sum = 0
        for j in range(m):
            inner_sum += lambdacg[j] * np.dot(Y[:, j].reshape(1, n), Y[:, j].reshape(n, 1))

        G = G - (1 / (2 * omega1)) * (2 * omega3 * ((1 / (2 * (2 * n * omega3 + 2 * omega1))) * inner_sum - n * omega4 * c / (2 * n * omega3 + 2 * omega1)) + omega4 * c) * np.identity(n)
        for j in range(m):
            G = G + 1 / (4 * omega1) * (lambdacg[j] * np.dot(Y[:, j].reshape(n, 1), Y[:, j].reshape(1, n)))


    else:  # Construct a full model
        
        b = F_values
        phi_Q = np.array([])
        for i in range(m):
            y = Y[:, i]
            y = y[np.newaxis]  
            aux_G = y * y.T - 0.5 * np.diag(pow(y, 2)[0])
            aux = np.array([])
            for j in range(n):
                aux = np.hstack((aux, aux_G[j:n, j]))

            phi_Q = np.vstack((phi_Q, aux)) if phi_Q.size else aux

        W = np.hstack((np.ones((m, 1)), Y.T))
        W = np.hstack((W, phi_Q))

        lambdacg = _obtain_coeffs(W, tol_svd, b, option='full')

        # g and G
        g = lambdacg[1:n + 1, :]
        cont = n + 1
        G = np.zeros((n, n))

        for j in range(n):
            G[j:n, j] = lambdacg[cont:cont + n - j, :].reshape((n - j,))
            cont = cont + n - j

        G = G + G.T - np.diag(np.diag(G))
    return (G, g, c)
