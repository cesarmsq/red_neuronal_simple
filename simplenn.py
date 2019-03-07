import numpy as np
from matplotlib import pyplot as plt

#funcion sigmoide
def sigmoid(x):
    return 1/(1+np.exp(-x))

#derivada de la funcion sigmoide
def dsigmoid(x):
    return x*(1-x)

#datos de entrenamiento (entrada)
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

#datos de entrenamiento (salida)
Y = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]]).T

#iniciando semilla y pesos aleatorios (-1 a 1)
np.random.seed(1)
pesos1 = 2*np.random.random((3,4)) - 1
pesos2 = 2*np.random.random((4,1)) - 1


#bucle de entrenamiento
for i in range(70000):
    #forward propagation
    l1 = sigmoid(X @ pesos1)
    salida = sigmoid(l1 @ pesos2)
    
    #error
    error = Y - salida
    
    #ajustes
    l2_delta = error * dsigmoid(salida)
    l1_delta = (l2_delta @ pesos2.T) * dsigmoid(l1)
    
    #actualizacion de los pesos
    pesos2 += (l1.T @ l2_delta)
    pesos1 += (X.T @ l1_delta)

#funcion para visualizar los datos
def vis_data():
    plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], c='b')
    plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], c='r')

#grafico del entrenamiento
for x in np.linspace(0,6,30):
    for y in np.linspace(0,3,30):
        c='b'
        r1 = sigmoid(np.array([[x,y,1]])@pesos1)
        r2 = sigmoid(r1 @ pesos2)
        if r2 >= 0.5:
            c='r'
        plt.scatter(x, y, c=c, alpha = 0.2)
vis_data()