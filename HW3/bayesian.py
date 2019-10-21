from matrix import Matrix
from generator import Linear 
import sys


b = float(sys.argv[1])
a = float(sys.argv[2])
w = [float(w) for w in sys.argv[3:]]
n = len(w)

px = []
py = []

linear = Linear(w, a)
I = Matrix(n)

s0 = I * b
m0 = Matrix([[0 for i in range(n)]]).transpose()
for i in range(1000):
    x, y = linear.gen()
    X = Matrix([[x**i for i in range(n)]])

    s1 = X.transpose()*X*(1.0/a) + s0
    m1 = s1.inverse() * (X.transpose()*(1.0/a)*y + s0*m0)

    ym = (X*m1).data[0][0]
    yv = a + (X*s1.inverse()*X.transpose()).data[0][0]
    print("x = %f, ym = %f, yv = %f, wm =" % (x, ym, yv), m1.transpose().data[0])

    s0, m0 = s1, m1

    if i == 999:
        import matplotlib.pyplot as plt
        import numpy as np
        import math
        px.append(x)
        py.append(y)
        X = np.linspace(-10, 10, num=1000)
        Y = []
        P = []
        U = []
        L = []
        for x in X:
            d = Matrix([[x**i for i in range(n)]])
            p = (d * m1).data[0][0]
            v = (1.0/a) + (d*s1.inverse()*d.transpose()).data[0][0]
            y = (d * Matrix([w]).transpose()).data[0][0]
            Y.append(y)
            P.append(p)
            U.append(p+math.sqrt(v))
            L.append(p-math.sqrt(v))
        plt.plot(X, U, c="c")           # predict + deviation
        plt.plot(X, L, c="c")           # predict - deviation
        plt.plot(X, P, c="b")           # predict line
        plt.plot(X, Y, c="r")           # true line
        plt.scatter(px, py, c="g")
        plt.show()