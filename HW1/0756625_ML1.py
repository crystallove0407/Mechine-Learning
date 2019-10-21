# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:25:40 2019

@author: cryst
"""
import numpy as np
import matplotlib.pyplot as plt

###input###
n = int(input("input n:"))
r = int(input("input lambda:"))

data = np.loadtxt("data.txt", delimiter=',')    #load data
x, b = np.hsplit(data, 2)


###LU decomposition###
def LUdecom(matrix):
    m, n = matrix.shape
    if m != n:
        return
    L = np.eye(n)
    U = matrix
    for i in range(n-1, 0, -1): 
        for j in range(1, i+1):  
            d = n-i-1   #對角項
            coef = U[j+d,d]/U[d,d]    
            L[j+d,d] = coef   #L
            for k in range(n):  #U
                if k >= d:
                    U[j+d, k] -= coef * U[d, k]
                else:
                    U[j+d, k] = 0
    return L, U

###To solve Ly=b Ux=y###
def solveLU(matrix, b):
    m, n = matrix.shape
    if m != n:
        return
    y = np.zeros(n)
    x = np.zeros(n)
    L, U = LUdecom(matrix)
    y[0] = b[0]                     #Ly=b
    for i in range(1, n):   
        colsum = 0
        for j in range(i):
            colsum += L[i][j] * y[j]
        y[i] = b[i] - colsum
        
    x[n-1] = y[n-1] / U[n-1, n-1]     #Uc=y
    for i in range(n-2, -1, -1):   
        colsum = 0
        for j in range(i+1, n):
            colsum += U[i, j] * x[j]
        x[i] = (y[i] - colsum) / U[i, i]
    return x


def yCalculate(xVal, res):
    yVal = res[n-1] * np.power(xVal, 0)
    for j in range(n-1):
        yVal = yVal + res[j] * np.power(xVal, n-1-j)
    return yVal


def TSEError(x, b, res, r):
    yVal = np.zeros((x.shape[0],1))
    for j in range(n):
        yVal = yVal + res[n-j-1] * np.power(x, j)
    error = np.power(np.abs(yVal-b), 2) + r * np.power(x, 2)
    error = np.sum(error)
    return error


def newtonError(x, b, res):
    yVal = np.zeros((x.shape[0],1))
    for j in range(n):
        yVal = yVal + res[n-j-1] * np.power(x, j)
    error = np.power(np.abs(yVal-b), 2)
    error = np.sum(error)
    return error


def printFunc(res, n):
    for i in range(n-1):
        print(res[i], "x^", n-i-1, " +", sep = '',end = ' ')
    print(res[n-1])

    
def LSE(n, r, x, b):
    ###build A###
    A = np.power(x, 0)
    for i in range(1, n):
        basis = np.power(x, i)
        A = np.concatenate((basis, A), axis=1)

    ###build AArI###
    I = np.eye(n)
    AArI = A.T.dot(A) + r * I
    Ab = A.T.dot(b)
    res = solveLU(AArI, Ab)
    totalError = TSEError(x, b, res, r)
    
    return res, totalError


def inverseMatrix(matrix):
    m, n = matrix.shape
    if m != n:
        return
    I = np.eye(n)
    inverse = []
    for i in I:
        inverse.append(solveLU(matrix, i.reshape(n, 1)))
    
    return np.array(inverse)


def newton(n, x, b, iteration = 2):
    ###build A###
    A = np.power(x, 0)
    for i in range(1, n):
        basis = np.power(x, i)
        A = np.concatenate((basis, A), axis=1)

    x_old = np.random.rand(n, 1)
    
    while True:
        if iteration < 0:
            break
        x_new = x_newton(A, x_old, b)
        x_old = x_new
        iteration -= 1
    
    totalError = newtonError(x, b, x_new)

    return x_new, totalError
    

def x_newton(A, x0, b):
    ###build f''=H ###
    inverseH = inverseMatrix(2*A.T.dot(A))
    
    ###build f'###
    f = 2*((A.T.dot(A)).dot(x0)) - 2*((A.T).dot(b))
    
    x1 = x0 - inverseH.dot(f)
    
    return x1


###show###
xVal = np.linspace(-6, 6, 60)
y_ticks = np.arange(0, 120, 20)

#LSE
LSE_res, LSE_err = LSE(n, r, x, b)
#outcome
print("LSE:")
print("Fitting line: ", end=" ")
printFunc(LSE_res, n)
print("Total error:", LSE_err)
print("")

plt.subplot(211)
plt.xlim(-6, 6)
plt.ylim(-20, 120)
plt.yticks(y_ticks)
plt.scatter(x, b, color="red")
plt.plot(xVal, yCalculate(xVal, LSE_res))

#Newton's method
newton_res, newton_err = newton(n, x, b)
#outcome
print("Newton's Method:")
print("Fitting line: ", end=" ")
printFunc(newton_res, n)
print("Total error:", newton_err)

plt.subplot(212)
plt.xlim(-6, 6)
plt.ylim(-20, 120)
plt.yticks(y_ticks)
plt.scatter(x, b, color="red")
plt.plot(xVal, yCalculate(xVal, newton_res))

plt.show()


    
    





