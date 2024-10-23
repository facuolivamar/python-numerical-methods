import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve, simplify

class CubicSpline:
    def __init__(self, X, Y):
        self.X = np.array(X, dtype=float)
        self.Y = np.array(Y, dtype=float)
        self.n = len(X) - 1  # Number of intervals
        self.h = np.diff(X)  # Step sizes
        self.coeffs = None  # Placeholder for coefficients
        self.x_symbol = symbols('x')  # Symbolic variable for polynomial

    def compute_splines(self):
        """
        Compute the coefficients for the cubic splines.
        """
        A = np.zeros((self.n + 1, self.n + 1))
        b = np.zeros(self.n + 1)

        # Natural boundary conditions (second derivative at the endpoints = 0)
        A[0, 0] = 1
        A[self.n, self.n] = 1

        # Fill matrix A and vector b with the conditions for internal points
        for i in range(1, self.n):
            A[i, i - 1] = self.h[i - 1]
            A[i, i] = 2 * (self.h[i - 1] + self.h[i])
            A[i, i + 1] = self.h[i]
            b[i] = 3 * ((self.Y[i + 1] - self.Y[i]) / self.h[i] -
                        (self.Y[i] - self.Y[i - 1]) / self.h[i - 1])

        # Solve the linear system for the second derivatives (c coefficients)
        c = np.linalg.solve(A, b)

        # Calculate a, b, and d coefficients
        a = self.Y[:-1]
        b = (self.Y[1:] - self.Y[:-1]) / self.h - self.h * (2 * c[:-1] + c[1:]) / 3
        d = (c[1:] - c[:-1]) / (3 * self.h)

        self.coeffs = np.array([a, b, c[:-1], d]).T  # Store coefficients

    def display_polynomials(self):
        """
        Display the cubic polynomials for each interval.
        """
        print("Cubic Polynomials for Each Interval:")
        for i in range(self.n):
            a, b, c, d = self.coeffs[i]
            polynomial = (a + 
                          b * (self.x_symbol - self.X[i]) + 
                          c * (self.x_symbol - self.X[i])**2 + 
                          d * (self.x_symbol - self.X[i])**3)
            print(f"S_{i}(x) = {simplify(polynomial)} for x in [{self.X[i]}, {self.X[i+1]}]")

    def evaluate(self, x):
        """
        Evaluate the spline at a given point x.
        """
        i = np.searchsorted(self.X, x) - 1
        i = min(max(i, 0), self.n - 1)  # Ensure valid index

        dx = x - self.X[i]
        a, b, c, d = self.coeffs[i]
        return a + b * dx + c * dx**2 + d * dx**3

    def plot(self):
        """
        Plot the original data points and the cubic spline.
        """
        X_interp = np.linspace(min(self.X), max(self.X), 500)
        Y_interp = [self.evaluate(x) for x in X_interp]

        plt.scatter(self.X, self.Y, color='red', label='Data Points')
        plt.plot(X_interp, Y_interp, label='Cubic Spline', color='blue')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Cubic Spline Interpolation')
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
    print(f"\nComputing cubic spline for dataset {i + 1}...")
    model = CubicSpline(X, Y)
    model.compute_splines()

    # Display the cubic polynomials
    model.display_polynomials()

    # Plot the data and the spline
    model.plot()
