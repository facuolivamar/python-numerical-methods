import numpy as np


class heuns():
    def __init__(self, f, y0, t0, t_end, h):
        self.f = f
        self.y0 = y0
        self.t0 = t0
        self.t_end = t_end
        self.h = h
        self.t_values, self.y_predictor, self.y_corrector = self.heuns_method(f, y0, t0, t_end, h)

    def heuns_method(self, f, y0, t0, t_end, h):
        n = int((t_end - t0) / h)
        t_values = np.linspace(t0, t_end, n+1)
        y_predictor = np.zeros(n+1)
        y_corrector = np.zeros(n+1)
        y_predictor[0] = y0
        y_corrector[0] = y0

        for i in range(1, n+1):
            # Predictor: Método de Euler
            y_pred = y_corrector[i-1] + h * f(t_values[i-1], y_corrector[i-1])
            y_predictor[i] = y_pred  # Guardamos el valor del predictor (Euler)

            # Corrector: Método de Heun
            y_corr = y_corrector[i-1] + (h / 2) * (f(t_values[i-1], y_corrector[i-1]) + f(t_values[i], y_pred))
            y_corrector[i] = y_corr  # Guardamos el valor del corrector (Heun)

        return t_values, y_predictor, y_corrector
    
    def get_values(self):
        return self.t_values, self.y_predictor, self.y_corrector
    
    def save_to_csv(self, filename):
        np.savetxt(filename, np.column_stack((self.t_values, self.y_predictor, self.y_corrector)), 
                   delimiter=",", header="t,y_predictor,y_corrector", comments="")

# Función f(x, y)
f = lambda x, y: np.exp(0.8 * x) - 0.5 * y

# Valores iniciales
y0, t0, t_end, h = 2, 0, 4, 0.1

# Ejemplo de uso del método de Heun
heun_example = heuns(f, y0, t0, t_end, h)

# Resultados
print("Resultados del Método de Heun (Predictor y Corrector):")
for t, y_pred, y_corr in zip(*heun_example.get_values()):
    print(f"x: {t:.4f}, y_pred: {y_pred:.4f}, y_corr: {y_corr:.4f}")

# Guardar resultados en CSV
heun_example.save_to_csv("heun_example.csv")
