import numpy as np
import numpy.linalg as LA

# Definición de la matriz A
A = np.array([1, 2, 1, 0, 1, 2, -1, 3, 2])
A = A.reshape((3, 3))

# Vector inicial
v = np.array([1, 1, 1], dtype=float)

maxIter = 10

for k in range(maxIter):
    # Resolver el sistema A*y = v para y (y = A^{-1}v)
    y = LA.solve(A, v)
    # Normalizar
    v = y / LA.norm(y)
    # Aproximación del valor propio más pequeño (inverso de la razón de Rayleigh)
    valprop = v.T @ A @ v
    print(f"Valor propio aproximado (potencia inversa): {valprop}")

# Comparación con los valores propios reales
print("Valores propios reales:", LA.eigvals(A))