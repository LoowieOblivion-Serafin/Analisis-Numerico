def metodo_potencia_inversa(A, mu, num_iteraciones=100, tol=1e-6):
    """
    Implementación del método de la potencia inversa para encontrar el valor propio
    más cercano a mu y el vector propio asociado.

    Args:
        A (np.ndarray): La matriz cuadrada para la cual se calcularán los valores propios.
        mu (float): El valor de desplazamiento alrededor del cual se buscará el valor propio.
        num_iteraciones (int): El número máximo de iteraciones.
        tol (float): La tolerancia para la convergencia.

    Returns:
        tuple: Una tupla que contiene:
            - lambda_aprox (float): El valor propio más cercano a mu.
            - v (np.ndarray): El vector propio asociado al valor propio más cercano a mu (normalizado).
    """
    n = A.shape[0]
    # Calcular la matriz (A - mu * I)
    B = A - mu * np.eye(n)

    # Factorización QR de B
    Q, R = factorQR(B)

    # Inicializar un vector aleatorio no nulo
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)  # Normalizar el vector inicial

    lambda_ant = 0

    for i in range(num_iteraciones):
        # Resolver (A - mu * I) * w = v para w
        # Esto es equivalente a B * w = v
        # Usamos la factorización QR: Q * R * w = v => R * w = Q.T * v
        y = Q.T @ v
        w = resuelve(R, y) # Utilizamos la función resuelve para resolver el sistema triangular superior

        # Normalizar el nuevo vector
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-12: # Evitar división por cero si w es casi cero
            print("Vector w es casi cero. El método de la potencia inversa puede fallar.")
            break
        v = w / w_norm

        # Calcular el valor propio de B, que es 1/sigma, donde sigma es el valor propio de B con mayor magnitud.
        # sigma está asociado a (A - mu * I) y 1/sigma a su inversa.
        # El valor propio de A más cercano a mu es mu + sigma_reciproco
        # El valor propio de B con mayor magnitud (sigma) es el valor propio dominante de B.
        # En el método de la potencia inversa aplicado a A-mu*I, estamos encontrando el valor propio de A-mu*I con la *menor* magnitud.
        # Sea lambda el valor propio de A-mu*I con la menor magnitud. Entonces (A-mu*I)*v = lambda*v.
        # Multiplicando por (A-mu*I)^(-1), tenemos v = lambda * (A-mu*I)^(-1) * v.
        # Esto implica que (A-mu*I)^(-1) * v = (1/lambda) * v.
        # Por lo tanto, 1/lambda es el valor propio dominante de (A-mu*I)^(-1).
        # El método de la potencia inversa busca el valor propio dominante de (A-mu*I)^(-1), que es 1/lambda.
        # El valor propio de A más cercano a mu es mu + lambda.
        # En cada iteración, el cociente de Rayleigh nos da una estimación de (1/lambda).
        # (v.T @ (A - mu * np.eye(n)) @ v) / (v.T @ v) da una estimación del valor propio de B (A-mu*I).
        # Queremos el valor propio de A, que es mu + (valor propio de B).
        # El método de la potencia aplicado a (A - mu*I)^-1 itera v_k+1 = (A-mu*I)^-1 * v_k.
        # El cociente de Rayleigh para (A-mu*I)^-1 es (v.T @ (A - mu * np.eye(n))^-1 @ v) / (v.T @ v).
        # Sin embargo, evitamos calcular la inversa explícitamente resolviendo el sistema lineal.
        # El valor propio dominante de (A-mu*I)^-1 converge a 1/lambda, donde lambda es el valor propio de A-mu*I con la menor magnitud.
        # Una mejor estimación del valor propio de A es el cociente de Rayleigh de A con el vector propio estimado v.
        lambda_actual = (v.T @ A @ v) / (v.T @ v)


        # Criterio de convergencia basado en el valor propio estimado
        if np.abs(lambda_actual - lambda_ant) < tol:
             print(f"Convergencia alcanzada en la iteración {i+1}")
             break
        lambda_ant = lambda_actual

    # El valor propio de A más cercano a mu es el último lambda_actual calculado.
    lambda_aprox = (v.T @ A @ v) / (v.T @ v)


    return lambda_aprox, v.flatten() # Removed [0][0] indexing

# Ejemplo de uso del método de la potencia inversa:
A = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]], dtype=float) # Asegurarse de que es float para la factorización QR

# Queremos encontrar el valor propio más cercano a 1.5
mu = 1.5

lambda_cercano, vector_propio_inversa = metodo_potencia_inversa(A, mu)

print("\n--- Método de la Potencia Inversa ---")
print(f"Valor propio más cercano a {mu}:", lambda_cercano)
print("Vector propio asociado:", vector_propio_inversa)

# Comprobación: A * v = lambda * v
print("\nComprobación (Potencia Inversa):")
print("A @ vector_propio_inversa:", A @ vector_propio_inversa)
print("lambda_cercano * vector_propio_inversa:", lambda_cercano * vector_propio_inversa)

# Comprobación con np.linalg.eigvals
print("\nValores propios de A (usando np.linalg.eigvals):", np.linalg.eigvals(A))
