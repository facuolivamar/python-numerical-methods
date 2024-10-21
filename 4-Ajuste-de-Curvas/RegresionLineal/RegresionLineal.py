import numpy as np
import matplotlib.pyplot as plt

# Datos
X = np.array([1, 2, 3, 4, 5, 6, 7])
Y = np.array([0.5, 2.5, 2.0, 4.0, 3.5, 6.0, 5.5])

# Cálculo de coeficientes de la regresión lineal
a1 = (np.sum(X * Y) - len(X) * np.mean(X) * np.mean(Y)) / (np.sum(X**2) - len(X) * np.mean(X)**2)
a0 = np.mean(Y) - a1 * np.mean(X)

# Predicción
Y_pred = a0 + a1 * X

# Visualización
plt.scatter(X, Y, color='red', label='Datos')
plt.plot(X, Y_pred, label=f'y = {a0:.4f} + {a1:.4f}x')
plt.legend()
plt.show()
