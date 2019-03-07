import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

X = np.array([ [3  , 1.5, 1],
               [2  , 0.5, 1],
               [3  , 1.3, 1],
               [1  , 1  , 1],
               [3.3, 1.4, 1],
               [1.5, 2  , 1],
               [2.8, 1.5, 1],
               [3  , 2.5, 1],
               [3.1, 1.7, 1],
               [4.5, 2.5, 1],
               [2.9, 1.2, 1],
               [4.5, 1.5, 1],
               [3.1, 1.7, 1],
               [3.5, 0.5, 1],
               [4.2, 2  , 1],
               [1.5, 0.7, 1],
               [2.3, 2.4, 1],
               [4.1, 1,   1]])

Y = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]]).T

np.random.seed(1)
pesos1 = 2*np.random.random((3,4)) - 1
pesos2 = 2*np.random.random((4,1)) - 1

costo = []
for i in range(70000):
    l1 = sigmoid(X @ pesos1)
    salida = sigmoid(l1 @ pesos2)
    
    error = Y - salida
    if (i % 700)==0:
        costo.append(np.mean(np.abs(error)))
    
    l2_delta = error * dsigmoid(salida)
    l1_delta = (l2_delta @ pesos2.T) * dsigmoid(l1)
    
    pesos2 += (l1.T @ l2_delta)
    pesos1 += (X.T @ l1_delta)
plt.plot(costo)