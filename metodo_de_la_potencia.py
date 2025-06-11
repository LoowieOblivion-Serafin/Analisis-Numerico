#Metodo de la potencia

import numpy.linalg as LA
A = np.array([1, 2, 1, 0, 1, 2, -1, 3, 2])
A = A.reshape((3,3))
v = np.array([1,1,1])
maxIter = 10
for k in range(maxIter):
    y = A@v
    v = y/LA.norm(y)
    valprop = v.T@A@v
    print(f"Valor propio dominante: {valprop}")
print(LA.eigvals(A))
