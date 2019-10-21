import math
import random

class Gaussian():
    def __init__(self, mean, var):
        self.mean = mean
        self.var  = var

    def gen(self):
        s = 1
        while s >= 1:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            s = x**2 + y**2
        x = x * math.sqrt(-2 * math.log(s) / s)
        y = y * math.sqrt(-2 * math.log(s) / s)
        return x * math.sqrt(self.var) + self.mean

class Linear():
    def __init__(self, w, a):
        self.w = w
        self.gaussian = Gaussian(0, a)

    def gen(self):
        x = random.uniform(-10, 10)
        r = 0
        for i, v in enumerate(self.w):
            r += x**i * v
        return x, r + self.gaussian.gen()

if __name__ == "__main__":
    import sys
    mode = int(sys.argv[1])
    if mode % 2 == 0:
        mean = float(sys.argv[2])
        var  = float(sys.argv[3])
        gaussian = Gaussian(mean, var)
        if mode / 2 == 0:
            print(gaussian.gen())
        else:
            import matplotlib.pyplot as plt
            X = [gaussian.gen() for i in range(10000)]
            plt.hist(X, bins=100)
            plt.show()
    else:
        a = float(sys.argv[2])
        w = [float(w) for w in sys.argv[3:]]
        linear = Linear(w, a)
        if mode / 2 == 0:
            print(linear.gen())
        else:
            import matplotlib.pyplot as plt
            X = []
            Y = []
            for i in range(10000):
                x, y = linear.gen()
                X.append(x)
                Y.append(y)

            plt.scatter(X, Y, s=1)
            plt.show()