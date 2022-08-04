"""
Description:
Codes for constructing least H^2 norm updating quadratic model 
Authors: Pengcheng Xie, Ya-xiang Yuan Connect: xpc@lsec.cc.ac.cn
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

    # Make sure the condition number is not too high
    indices = S < tol_svd
    S[indices] = tol_svd
    Sinv = np.diag(1 / S)
    V = VT.T
    # Get the coefficients
    lambda_0 = reduce(np.dot, [V, Sinv, U.T, b])
    return (lambda_0)


def quad_Frob(X, F_values, args):
    """
    Given a set of points in the trust region
    and their values, construct a quadratic model
    in the form of g.T x + 1/2 x.T H x + \alpha.

    If the number of points are less than
    (n+1) (n+2)/2 then build the model such that the
    Frobenius norm is minimized. In this code the KKT
    conditions are solved. Otherwise, solve
    the system of equations used in polynomial interpolation.
    M(\phi, Y) \lambda = f

    arguments: X: the sample points to be interpolated
        F_values: the corresponding true solutions to the sample points
    outputs: g and H in the quadratic model
    """
    # Minimum value accepted for a singular value
    eps = np.finfo(float).eps
    tol_svd = eps ** 5
    # n = number of variables m = number of points
    (n, m) = X.shape

    H = np.zeros((n, n))
    g = np.zeros((n, 1))

    # Shift the points to the origin
    Y = X - np.dot(np.diag(X[:, 0]), np.ones((n, m)))

    if (m < (n + 1) * (n + 2) / 2):
        c1 = args.c1
        c2 = args.c2
        c3 = args.c2
        r = 1
        V2 = (np.pi ** (n / 2) / math.gamma(n / 2 + 1))

        
        omega1 = (c1 * r ** 4 / (2 * (n + 4) * (n + 2)) + c2 * r ** 2 / (n + 2) + c3)
        omega2 = (c1 * r ** 2 / (n + 2) + c2)
        omega3 = c1 * r ** 4 / (4 * (n + 4) * (n + 2))
        omega4 = c1 * r ** 2 / (n + 2)
        omega5 = c1
        

       
        b = np.vstack((F_values, np.zeros((n + 1, 1))))
       
        A = 1 / (8 * omega1) * (np.dot(Y.T, Y) ** 2)
        for i in range(len(A)):
            for j in range(len(A)):
                A[i, j] = A[i, j] - (omega3 / (8 * omega1 * (n * omega3 + omega1))) * np.dot(Y[:, i].T,
                                                                                             Y[:, i]) * np.dot(
                    Y[:, j].T, Y[:, j])

        J = np.zeros((m, 1))  # 式（2）J
        # X_bar = np.zeros((1, m))  # 式（2） Y^T
        # B1 = np.zeros((1, m))  # 式（2）B
        for i in range(m):
            J[i] = 1 - omega4 / (4 * omega1 + 4 * n * omega3) * np.dot(Y[:, i].T, Y[:, i])
        

        line1 = np.hstack((A, J, Y.T))  
        line2 = np.hstack((J.T, ((n * omega4 ** 2) / (2 * n * omega3 + 2 * omega1) - 2 * omega5) * np.ones((1, 1)),
                           np.zeros((n, 1)).T))
        line3 = np.hstack((Y, np.zeros((n, 1)), -2 * omega2 * np.identity(n)))

        W = np.vstack((line1, line2, line3))

        lambda_0 = _compute_coeffs(W, tol_svd, b, option='partial')  

        # Grab the coeffs of linear terms (g) and the ones of quadratic terms
        # (H) for g.T s + s.T H s
        c = lambda_0[m:m + 1]
        g = lambda_0[m + 1:m + 1 + n]
        # T = lambda_0[m + 1 + n]

        H = np.zeros((n, n))  
        
        inner_sum = 0
        for j in range(m):
            inner_sum += lambda_0[j] * np.dot(Y[:, j].reshape(1, n), Y[:, j].reshape(n, 1))

        H = H - (1 / (2 * omega1)) * (2 * omega3 * (
                    (1 / (2 * (2 * n * omega3 + 2 * omega1))) * inner_sum - n * omega4 * c / (
                        2 * n * omega3 + 2 * omega1)) + omega4 * c) * np.identity(n)
        for j in range(m):
            H = H + 1 / (4 * omega1) * (lambda_0[j] * np.dot(Y[:, j].reshape(n, 1), Y[:, j].reshape(1, n)))


    else:  # Construct a full model
        # Here we have enough points. Solve the sys of equations.
        b = F_values
        phi_Q = np.array([])
        for i in range(m):
            y = Y[:, i]
            y = y[np.newaxis]  # turn y from 1D to a 2D array
            aux_H = y * y.T - 0.5 * np.diag(pow(y, 2)[0])
            aux = np.array([])
            for j in range(n):
                aux = np.hstack((aux, aux_H[j:n, j]))

            phi_Q = np.vstack((phi_Q, aux)) if phi_Q.size else aux

        W = np.hstack((np.ones((m, 1)), Y.T))
        W = np.hstack((W, phi_Q))

        lambda_0 = _compute_coeffs(W, tol_svd, b, option='full')

        # Retrieve the model coeffs (g) and (H)
        g = lambda_0[1:n + 1, :]
        cont = n + 1
        H = np.zeros((n, n))

        for j in range(n):
            H[j:n, j] = lambda_0[cont:cont + n - j, :].reshape((n - j,))
            cont = cont + n - j

        H = H + H.T - np.diag(np.diag(H))
    return (H, g, c)
