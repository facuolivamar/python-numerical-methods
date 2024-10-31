import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import symbols, simplify, prod

class LagrangeInterpolation:
    def __init__(self, X, Y, title):
        """
        Initialize the LagrangeInterpolation model with input data X and Y.
        """
        self.X = np.array(X, dtype=float)
        self.Y = np.array(Y, dtype=float)
        self.n = len(X)
        self.title = title  # Title for the plot
        self.x_symbol = symbols('x')  # Symbolic variable for polynomial
        self.basis_polynomials = []  # Store basis polynomials L_i(x)
        self.iterated_polynomials = []  # Store polynomial expressions iteratively

    def lagrange_basis(self, i):
        """
        Constructs the i-th Lagrange basis polynomial symbolically following the structure.
        """
        numerator = prod((self.x_symbol - self.X[j]) for j in range(self.n) if j != i)
        denominator = prod((self.X[i] - self.X[j]) for j in range(self.n) if j != i)
        basis = numerator / denominator  # L_i(x) = Π(x - x_j) / Π(x_i - x_j)

        # Store the constructed basis polynomial L_i(x)
        self.basis_polynomials.append(simplify(basis))
        return simplify(basis)

    def construct_polynomial(self):
        """
        Constructs the full Lagrange interpolating polynomial.
        """
        polynomial = 0  # Initialize polynomial

        # Construct the polynomial iteratively using the Lagrange basis polynomials
        for i in range(self.n):
            term = self.Y[i] * self.lagrange_basis(i)
            polynomial += term  # Add the term to the overall polynomial
            self.iterated_polynomials.append(simplify(polynomial))  # Log each iteration

        return simplify(polynomial)  # Return the final simplified polynomial

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

    def save_to_excel(self, filename):
        """
        Save the basis polynomials and iterated polynomials to an Excel file.
        """
        # Create DataFrames for the basis and iterated polynomials
        df_basis = pd.DataFrame({
            f"L_{i}(x)": [str(basis)] for i, basis in enumerate(self.basis_polynomials)
        })
        df_polynomials = pd.DataFrame({
            "Polynomial Iteration": [str(poly) for poly in self.iterated_polynomials]
        })

        # Save to Excel with appropriate sheet names
        with pd.ExcelWriter(filename) as writer:
            df_basis.to_excel(writer, sheet_name='Basis Polynomials', index=False)
            df_polynomials.to_excel(writer, sheet_name='Iterated Polynomials', index=False)

        print(f"Saved polynomial iterations to {filename}")

    def plot(self, polynomial):
        """
        Plot the original data points and the interpolated polynomial.
        """
        X_interp = np.linspace(min(self.X), max(self.X), 100)
        Y_interp = [self.interpolate(x) for x in X_interp]

        plt.scatter(self.X, self.Y, color='red', label='Data Points')
        plt.plot(X_interp, Y_interp, label='Interpolated Polynomial', color='blue')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Lagrange Interpolation - {self.title}')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage with multiple datasets
datasets = [
    # ([0.0, 1.0, 2.0, 3.0], [1.0, 2.7182, 7.3891, 20.0855], 'Dataset 1'),
    # ([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [1.0, 2.1190, 2.9100, 3.9450, 5.7200, 8.6950], 'Dataset 2'),
    # ([1.0, 2.0, 3.0, 5.0, 6.0], [4.75, 4.0, 5.25, 19.75, 36.0], 'Dataset 3'),
    # ([0, 10, 20, 30, 40, 50, 60], [50000, 35000, 31000, 20000, 19000, 12000, 11000], 'Dataset 4'),
    ([-1.0000, 0.0000, 1.0000, 2.0000], [2.0000, -0.7183, 0.0000, 0.8964], 'Dataset 5')
]

# Process each dataset and save the results to Excel
for i, (X, Y, title) in enumerate(datasets):
    model = LagrangeInterpolation(X, Y, title)

    # Construct and display the polynomial
    polynomial = model.construct_polynomial()
    print(f"Interpolated Polynomial for {title}: {polynomial}")

    # Plot the data and the polynomial
    model.plot(polynomial)

    # Save the polynomial iterations and basis polynomials to Excel
    model.save_to_excel(f'lagrange_interpolation_{i+1}.xlsx')

other_plots = [-1.5342, -0.5732, 1.5000, 1.5674]

def polynomial_interpolation(x):
    return -0.543083333333333*(x**3)+ 1.7183*(x**2) - 0.456916666666667*x - 0.7183

for x in other_plots:
    print(f"Interpolated value at {x}: {polynomial_interpolation(x)}")
    