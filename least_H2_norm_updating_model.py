"""
Codes for constructing least H^2 norm updating quadratic model
Authors: Pengcheng Xie, Ya-xiang Yuan
Connect: xpc@lsec.cc.ac.cn
"""
import numpy as np
from functools import reduce
import math
from scipy import linalg as LA

def _compute_coeffs(W, tol_svd, b, option):

    if option == 'partial':
        U, S, VT = LA.svd(W)
    else:
        U, S, VT = LA.svd(W, full_matrices=False)

    indices = S < tol_svd
    S[indices] = tol_svd
    Sinv = np.diag(1/S)
    V = VT.T
    # Get the coefficients
    lambda_0 = reduce(np.dot, [V, Sinv, U.T, b])
    return (lambda_0)

def quad_Frob(X, F_values, c1, c2, c3):
    """
    arguments: 
    X: the sample points to be interpolated
    F_values: the corresponding true solutions to the points
    """
    eps = np.finfo(float).eps
    tol_svd = eps**5
    # n = number of variables 
    # m = number of points
    (n, m) = X.shape

    H = np.zeros((n, n))
    g = np.zeros((n, 1))

    Y = X - np.dot(np.diag(X[:, 0]), np.ones((n, m)))

    if (m < (n+1)*(n+2)/2):
        r = 1
        #V2 = (np.pi**(n/2)/math.gamma(n/2+1))
        omega1 = (c1*r**4/(2*(n+4)*(n+2))+c2*r**2/(n+2)+c3)
        omega2 = (c1*r**2/(n+2)+c2)
        omega3 = c1*r**(4)/(4*(n+4)*(n+2))
        omega4 = c1*r**(2)/(n+2)
        omega5 = c1
        # the solution of the KKT conditions
        b = np.vstack((F_values, np.zeros((n+2, 1))))
        A = 1/(8*omega1) * (np.dot(Y.T, Y)**2)
        J = np.zeros((m,1))
        X_bar = np.zeros((1,m))
        B1 = np.zeros((1,m))
        for i in range(m):
            J[i] = 1 - omega4/(4*omega1)*np.dot(Y[:,i].T,Y[:,i])
            X_bar[0,i] = -omega3/(2*omega1)*np.dot(Y[:,i].T,Y[:,i])
            B1[0,i] = 0.5*np.dot(Y[:,i].T,Y[:,i])

        # Construct W 
        line1 = np.hstack((A, np.ones((m, 1)), Y.T, X_bar.T))
        line2 = np.hstack((np.ones((1,m)), -2*omega5*np.ones((1,1)), np.zeros((1,n)), -omega4*np.ones((1,1))))
        line3 = np.hstack((Y, np.zeros((n,1)), -2*omega2*np.identity(n), np.zeros((n,1))))
        line4 = np.hstack((B1, -n*omega4*np.ones((1,1)), np.zeros((1,n)), (-2*n*omega3-2*omega1)*np.ones((1,1))))
        W = np.vstack((line1,line2,line3,line4))
        lambda_0 = _compute_coeffs(W, tol_svd, b, option='partial')

        c = lambda_0[m:m+1]
        g = lambda_0[m+1:m+1+n]
        T = lambda_0[m+1+n]
        
        H = np.zeros((n, n))
        for j in range(m):
            H = H + 1/(4*omega1)*(lambda_0[j] *
                    np.dot(Y[:, j].reshape(n, 1), Y[:, j].reshape(1, n)) 
                    - (2*omega3*T+omega4*c)*np.identity(n))

    else:  # Construct a full model
        b = F_values
        phi_Q = np.array([])
        for i in range(m):
            y = Y[:, i]
            y = y[np.newaxis]
            aux_H = y * y.T - 0.5 * np.diag(pow(y, 2)[0])
            aux = np.array([])
            for j in range(n):
                aux = np.hstack((aux, aux_H[j:n, j]))

            phi_Q = np.vstack((phi_Q, aux)) if phi_Q.size else aux

        W = np.hstack((np.ones((m, 1)), Y.T))
        W = np.hstack((W, phi_Q))

        lambda_0 = _compute_coeffs(W, tol_svd, b, option='full')

        g = lambda_0[1:n+1, :]
        cont = n+1
        H = np.zeros((n, n))

        for j in range(n):
            H[j:n, j] = lambda_0[cont:cont + n - j, :].reshape((n-j,))
            cont = cont + n - j

        H = H + H.T - np.diag(np.diag(H))
    return (H, g)
