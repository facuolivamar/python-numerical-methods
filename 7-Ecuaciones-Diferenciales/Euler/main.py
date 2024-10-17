import numpy as np
"""
 ir de 0.1 en 0.1 hasta llegar a 4
 y1 = y0 + h * f(t0, y0)
 y1 = y0 + (f(x0)-0.5*y0)*h
"""

def eulers_method(f, y0, t0, t_end, h):
    n = int((t_end - t0) / h)
    t_values = np.linspace(t0, t_end, n+1)
    y_values = np.zeros(n+1)
    y_values[0] = y0
    for i in range(1, n+1):
        y_values[i] = y_values[i-1] + h * f(t_values[i-1], y_values[i-1])
    return t_values, y_values

# Example usage from pdf pptx
f = lambda x, y: np.exp(0.8 * x) - 0.5 * y
y0, t0, t_end, h = 2, 0, 4, 0.1
t_values, y_values = eulers_method(f, y0, t0, t_end, h)
for value in zip(t_values, y_values):
    print(value)

# save to a csv file
np.savetxt("euler.csv", np.column_stack((t_values, y_values)), delimiter=",", header="t,y", comments="")
