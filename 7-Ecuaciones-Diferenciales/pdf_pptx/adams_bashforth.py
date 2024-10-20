# adams_bashforth.py
import numpy as np

def adams_bashforth_4(f, t0, y0, h, n):
    """
    Implementación del método de Adams-Bashforth de orden 4.
    
    Parámetros:
    f  : función que define la EDO, f(t, y)
    t0 : valor inicial del tiempo
    y0 : valor inicial de y
    h  : tamaño del paso
    n  : número de pasos
    
    Retorna:
    t_vals: valores de tiempo
    y_vals: soluciones aproximadas de y en cada paso
    """
    # Inicializamos los arrays para almacenar los resultados
    t_vals = np.zeros(n+1)
    y_vals = np.zeros(n+1)

    # Condiciones iniciales
    t_vals[0] = t0
    y_vals[0] = y0

    # Primeros 3 pasos con otro método (por ejemplo, Runge-Kutta de orden 4)
    for i in range(3):
        k1 = h * f(t_vals[i], y_vals[i])
        k2 = h * f(t_vals[i] + h / 2, y_vals[i] + k1 / 2)
        k3 = h * f(t_vals[i] + h / 2, y_vals[i] + k2 / 2)
        k4 = h * f(t_vals[i] + h, y_vals[i] + k3)
        y_vals[i+1] = y_vals[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        t_vals[i+1] = t_vals[i] + h

    # Coeficientes del método Adams-Bashforth de orden 4
    beta = [55 / 24, -59 / 24, 37 / 24, -9 / 24]

    # Aplicamos Adams-Bashforth para los siguientes pasos
    for i in range(3, n):
        y_vals[i+1] = (y_vals[i] + h * (
            beta[0] * f(t_vals[i], y_vals[i]) +
            beta[1] * f(t_vals[i-1], y_vals[i-1]) +
            beta[2] * f(t_vals[i-2], y_vals[i-2]) +
            beta[3] * f(t_vals[i-3], y_vals[i-3])
        ))
        t_vals[i+1] = t_vals[i] + h

    return t_vals, y_vals

# Ejemplo de uso
def f(t, y):
    return t - y  # Definimos la EDO dy/dt = t - y

# Parámetros
t0 = 0
y0 = 1
h = 0.1
n = 10

t_vals, y_vals = adams_bashforth_4(f, t0, y0, h, n)

# Imprimimos los resultados
for t, y in zip(t_vals, y_vals):
    print(f"t = {t:.2f}, y = {y:.5f}")
