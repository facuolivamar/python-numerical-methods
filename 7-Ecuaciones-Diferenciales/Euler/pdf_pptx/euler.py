import numpy as np
"""
 ir de 0.1 en 0.1 hasta llegar a 4
 y1 = y0 + h * f(t0, y0)
 y1 = y0 + (f(x0)-0.5*y0)*h
"""
class eulers():
    def __init__(self, f, y0, t0, t_end, h):
        self.f = f
        self.y0 = y0
        self.t0 = t0
        self.t_end = t_end
        self.h = h
        self.t_values, self.y_values = self.eulers_method(f, y0, t0, t_end, h)

    def eulers_method(self, f, y0, t0, t_end, h):
        n = int((t_end - t0) / h)
        t_values = np.linspace(t0, t_end, n+1)
        y_values = np.zeros(n+1)
        y_values[0] = y0
        for i in range(1, n+1):
            y_values[i] = y_values[i-1] + h * f(t_values[i-1], y_values[i-1])
        return t_values, y_values
    
    def get_values(self):
        return self.t_values, self.y_values
    
    def save_to_csv(self, filename):
        np.savetxt(filename, np.column_stack((self.t_values, self.y_values)), delimiter=",", header="t,y", comments="")

# pptx_8_2: Example usage from pdf pptx
f = lambda x, y: np.exp(0.8 * x) - 0.5 * y
y0, t0, t_end, h = 2, 0, 4, 0.1
euler_example = eulers(f, y0, t0, t_end, h)

# Resultados
print("Resultados del Método de Heun (Predictor y Corrector):")
for t_val, y_val in zip(*euler_example.get_values()):
    print(f"x: {t_val:.4f}, y_pred: {y_val:.4f}")

euler_example.save_to_csv("euler_example.csv")
