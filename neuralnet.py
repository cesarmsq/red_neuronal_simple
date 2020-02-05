import numpy as np
import matplotlib.pyplot as plt

sigmoid = lambda x: 1/(1+np.exp(-x))
dsigmoid= lambda x: x*(1-x)

X = np.array([[3  , 1.5, 1],
              [2  ,   1, 1],
              [4  , 1.5, 1],
              [3  ,   1, 1],
              [3.5, 0.5, 1],
              [2  , 0.5, 1],
              [5.5,   1, 1],
              [1  ,   1, 1]])

y = np.array([[1],
              [0],
              [1],
              [0],
              [1],
              [0],
              [1],
              [0]])

# dim W1 = entradas x neuronas en capa oculta (3 entrada x 5 capa oculta)
W1 = 2 * np.random.random((3, 5)) - 1
# dim W2 = neuronas en capa oculta+1(bias) x salidas (5+1capa oculta x 1 salida)
W2 = 2 * np.random.random((6, 1)) - 1

errores = []
for i in range(15000):
    z1 = X@W1
    fz1 = np.hstack((sigmoid(z1), np.ones((X.shape[0], 1))))

    z2 = fz1@W2
    y_hat = sigmoid(z2)

    error = 0.5*(y-y_hat)**2
    errores.append(error.sum())

    delta3 = (-(y-y_hat)) * dsigmoid(y_hat)
    djdw2  = fz1.T@delta3

    delta2= (delta3@W2.T) * dsigmoid(fz1)
    djdw1 = X.T@delta2

    W2 -= 0.1*djdw2
    W1 -= 0.1*djdw1[:,:1]

print(f"Error: {error.sum()*100} %")
print(f"Precisión: {(1-error.sum())*100} %")
plt.plot(errores)
plt.show()
