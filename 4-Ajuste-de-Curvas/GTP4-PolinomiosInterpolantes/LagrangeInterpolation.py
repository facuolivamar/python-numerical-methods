import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, simplify, prod

class LagrangeInterpolation:
    def __init__(self, X, Y):
        self.X = np.array(X, dtype=float)
        self.Y = np.array(Y, dtype=float)
        self.n = len(X)
        self.x_symbol = symbols('x')  # Symbolic variable for polynomial

    def lagrange_basis(self, i):
        """
        Constructs the i-th Lagrange basis polynomial symbolically.
        """
        basis = prod((self.x_symbol - self.X[j]) / (self.X[i] - self.X[j])
                     for j in range(self.n) if j != i)
        return simplify(basis)

    def construct_polynomial(self):
        """
        Constructs the full Lagrange interpolating polynomial.
        """
        polynomial = sum(self.Y[i] * self.lagrange_basis(i) for i in range(self.n))
        return simplify(polynomial)

    def interpolate(self, x):
        """
        Evaluates the Lagrange polynomial at a given point x.
        """
        result = 0
        for i in range(self.n):
            term = self.Y[i]
            for j in range(self.n):
                if i != j:
                    term *= (x - self.X[j]) / (self.X[i] - self.X[j])
            result += term
        return result

    def plot(self, polynomial=None):
        """
        Plots the original data points and the interpolated polynomial.
        """
        X_interp = np.linspace(min(self.X), max(self.X), 100)
        Y_interp = [self.interpolate(x) for x in X_interp]

        plt.scatter(self.X, self.Y, color='red', label='Data Points')
        plt.plot(X_interp, Y_interp, label='Interpolated Polynomial', color='blue')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Lagrange Interpolation {polynomial}')
        plt.legend()
        plt.grid(True)
        plt.show()

# Usage for each dataset
datasets = [
    ([0.0, 1.0, 2.0, 3.0], [1.0, 2.7182, 7.3891, 20.0855]),  # Dataset Ejercicio 1.1
    ([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [1.0, 2.1190, 2.9100, 3.9450, 5.7200, 8.6950]),  # Dataset Ejercicio 1.2
    ([1.0, 2.0, 3.0, 5.0, 6.0], [4.75, 4.0, 5.25, 19.75, 36.0]),  # Dataset Ejercicio 1.3
    ([0, 10, 20, 30, 40, 50, 60], [50000, 35000, 31000, 20000, 19000, 12000, 11000])  # Dataset Ejercicio 1.4
]

for i, (X, Y) in enumerate(datasets):
    print(f"\nInterpolating dataset {i + 1} using Lagrange's method...")
    model = LagrangeInterpolation(X, Y)
    
    # Construct and display the polynomial
    polynomial = model.construct_polynomial()
    print(f"Interpolated Polynomial for dataset {i + 1}: {polynomial}")

    # Plot the data and the polynomial
    model.plot(polynomial=polynomial)
