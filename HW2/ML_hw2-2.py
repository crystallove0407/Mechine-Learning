import numpy as np
from math import factorial

def read(file):
    datas = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            datas.append(np.array(line))
    return datas

def Beta(a,b):
    return (Gamma(a) * Gamma(b)) / Gamma(a + b)

def Gamma(x):
    return factorial(x - 1)

def Likelihood(p, m, N):
    return (factorial(N)/(factorial(m)*factorial(N-m))) * (p**m) * ((1-p)**(N-m)) 


datas = read(input("file name:"))
beta_a = int(input("beta_a:"))
beta_b = int(input("beta_b:"))


count0 = np.char.count(datas, '0')
count1 = np.char.count(datas, '1')

for idx, (c0, c1) in enumerate(zip(count0, count1)):
    prior_a = beta_a
    prior_b = beta_b
    beta_a += c1
    beta_b += c0
    likelihood = Likelihood(c1/(c0+c1) , c1, c0+c1)
    
    print("case {}: {}".format(idx+1, datas[idx]))
    print("Likelihood: {}".format(likelihood))
    print("Beta prior:     a = {:<4d}b = {:<4d}".format(prior_a, prior_b))
    print("Beta posterior: a = {:<4d}b = {:<4d}\n".format(beta_a, beta_b))
