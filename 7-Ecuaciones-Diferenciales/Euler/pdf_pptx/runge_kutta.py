import numpy as np

def runge_kutta_2(f, y0, t0, t_end, h):
    n = int((t_end - t0) / h)
    t_values = np.linspace(t0, t_end, n+1)
    y_values = np.zeros(n+1)
    y_values[0] = y0
    k1_values = np.zeros(n+1)
    k2_values = np.zeros(n+1)

    for i in range(1, n+1):
        t = t_values[i-1]
        y = y_values[i-1]

        # K1 y K2 para el método de Runge-Kutta de segundo orden
        k1 = h * f(t, y)
        k2 = h * f(t + h, y + k1)
        
        # Actualizar el valor de y para el siguiente paso
        y_values[i] = y + (0.5 * k1 + 0.5 * k2)
        k1_values[i-1] = k1
        k2_values[i-1] = k2

    return t_values, y_values, k1_values, k2_values

# Ejemplo de uso
# Definir una función f(t, y)
def f(t, y):
    return np.exp(0.8 * t) - 0.5 * y  # Esto es solo un ejemplo, puedes cambiar la función

# Condiciones iniciales
y0 = 2   # valor inicial de y
t0 = 0   # tiempo inicial
t_end = 4  # tiempo final
h = 0.1  # tamaño del paso

# Ejecutar el método de Runge-Kutta de segundo orden
t_values, y_values, k1_values, k2_values = runge_kutta_2(f, y0, t0, t_end, h)

# Imprimir resultados
for t, y, k1, k2 in zip(t_values, y_values, k1_values, k2_values):
    print(f"t: {t:.8f}, y: {y:.8f} k1: {k1:.8f} k2: {k2:.8f}")

print(f"Resultado final de Runge-Kutta 2do orden: {y_values[-1]}")

np.savetxt("rungeKutta_example.csv", np.column_stack((t_values, y_values, k1_values, k2_values)), 
                   delimiter=",", header="t, y, k1, k2", comments="")

