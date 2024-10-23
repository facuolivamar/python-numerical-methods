import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, simplify

class NewtonInterpolation:
    def __init__(self, X, Y):
        self.X = np.array(X, dtype=float)
        self.Y = np.array(Y, dtype=float)
        self.n = len(X)
        self.coefficients = np.zeros((self.n, self.n))
        self.x_symbol = symbols('x')  # Symbolic variable for polynomial

    def divided_difference_table(self):
        """
        Builds the divided difference table and extracts the coefficients.
        """
        self.coefficients[:, 0] = self.Y  # First column is the Y values

        # Calculate divided differences
        for j in range(1, self.n):
            for i in range(self.n - j):
                self.coefficients[i, j] = (self.coefficients[i + 1, j - 1] - self.coefficients[i, j - 1]) / (self.X[i + j] - self.X[i])

        return self.coefficients[0, :]  # Return the top row (coefficients)

    def construct_polynomial(self):
        """
        Construct the interpolated polynomial symbolically.
        """
        polynomial = 0
        term = 1

        # Build the polynomial using the divided difference coefficients
        for i in range(self.n):
            polynomial += self.coefficients[0, i] * term
            if i < self.n - 1:
                term *= (self.x_symbol - self.X[i])

        return simplify(polynomial)  # Simplify the polynomial expression

    def interpolate(self, x):
        """
        Evaluate the polynomial at a given point x.
        """
        result = self.coefficients[0, 0]
        term = 1.0
        for i in range(1, self.n):
            term *= (x - self.X[i - 1])
            result += self.coefficients[0, i] * term
        return result

    def plot(self, polynomial=None):
        """
        Plot the original data points and the interpolated polynomial.
        """
        X_interp = np.linspace(min(self.X), max(self.X), 100)
        Y_interp = [self.interpolate(x) for x in X_interp]

        plt.scatter(self.X, self.Y, color='red', label='Data Points')
        plt.plot(X_interp, Y_interp, label='Interpolated Polynomial', color='blue')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Newton Interpolation {polynomial}')
        plt.legend()
        plt.grid(True)
        plt.show()

# Usage for each dataset
datasets = [
    ([0.0, 1.0, 2.0, 3.0], [1.0, 2.7182, 7.3891, 20.0855]),  # Dataset Ejercicio 1.1
    ([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [1.0, 2.1190, 2.9100, 3.9450, 5.7200, 8.6950]),  # Dataset Ejercicio 1.2
    ([1.0, 2.0, 3.0, 5.0, 6.0], [4.75, 4.0, 5.25, 19.75, 36.0]),  # Dataset Ejercicio 1.3
    ([0, 10, 20, 30, 40, 50, 60], [50000, 35000, 31000, 20000, 19000, 12000, 11000])  # Dataset Ejercicio 1.3
]

for i, (X, Y) in enumerate(datasets):
    print(f"\nInterpolating dataset {i + 1}...")
    model = NewtonInterpolation(X, Y)
    model.divided_difference_table()
    
    # Construct and display the polynomial
    polynomial = model.construct_polynomial()
    print(f"Interpolated Polynomial for dataset {i + 1}: {polynomial}")

    # Plot the data and the polynomial
    model.plot(polynomial=polynomial)
