import numpy as np

class rungeKutta():
    def __init__(self, f, y0, t0, t_end, h, orden):
        self.f = f
        self.y0 = y0
        self.t0 = t0
        self.t_end = t_end
        self.h = h
        self.orden = orden
        self.t_values, self.y_values = self.runge_kutta(f, y0, t0, t_end, h)

    def runge_kutta(self, f, y0, t0, t_end, h):
        n = int((t_end - t0) / h)
        t_values = np.linspace(t0, t_end, n+1)
        y_values = np.zeros(n+1)
        y_values[0] = y0

        for i in range(1, n+1):
            t = t_values[i-1]
            y = y_values[i-1]
            
            if self.orden == 2:
                # K1 y K2 para el método de Runge-Kutta de segundo orden
                k1 = f(t, y)
                k2 = f(t + h, y + h*k1)

                # Actualizar el valor de y para el siguiente paso
                y_values[i] = y + (0.5 * k1 + 0.5 * k2)*h
            elif self.orden == 3:
                k1 = f(t, y)
                k2 = f(t + 0.5*h, y + 0.5*h*k1)
                k3 = f(t + 0.5*h, y - h*k1 + 2*h*k2)

                y_values[i] = y + ((k1 + 4*k2 + k3)/6)*h
            elif self.orden == 4:
                k1 = f(t, y)
                k2 = f(t + 0.5*h, y + 0.5*h*k1)
                k3 = f(t + 0.5*h, y + 0.5*h*k2)
                k4 = f(t + h, y + h*k3)

                y_values[i] = y + ((k1 + 2*k2 + 2*k3 + k4)/6)*h
            else:
                pass
                

        return t_values, y_values

    def get_values(self):
        return self.t_values, self.y_values


    def save_to_csv(self, filename):
        np.savetxt(filename, np.column_stack((self.t_values, self.y_values)), 
                   delimiter=",", header="t, y", comments="")


# Ejemplo de uso
f = lambda x, y: np.exp(0.8 * x) - 0.5 * y

# Condiciones iniciales
y0 = 2   # valor inicial de y
t0 = 0   # tiempo inicial
t_end = 4  # tiempo final
h = 0.1  # tamaño del paso

# Ejecutar el método de Runge-Kutta de segundo orden
rungeKutta2_example = rungeKutta(f, y0, t0, t_end, h, 2)
t_values, y_values = rungeKutta2_example.get_values()

# Imprimir resultados
for t, y in zip(t_values, y_values):
    print(f"t: {t:.8f}, y: {y:.8f}")

print(f"Resultado final de Runge-Kutta 2do orden: {y_values[-1]}")

rungeKutta2_example.save_to_csv("rungeKutta2_example.csv")

# 3er Orden
rungeKutta3_example = rungeKutta(f, y0, t0, t_end, h, 3)
t_values3, y_values3 = rungeKutta3_example.get_values()
print(f"Resultado final de Runge-Kutta 3er orden: {y_values3[-1]}")

# 4to Orden
rungeKutta4_example = rungeKutta(f, y0, t0, t_end, h, 4)
t_values4, y_values4 = rungeKutta4_example.get_values()
print(f"Resultado final de Runge-Kutta 4to orden: {y_values4[-1]}")
