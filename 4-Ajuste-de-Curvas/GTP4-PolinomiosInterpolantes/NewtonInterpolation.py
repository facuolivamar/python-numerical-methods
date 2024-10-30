import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import symbols, simplify, lambdify

class NewtonInterpolation:
    def __init__(self, X, Y, title):
        """
        Initialize the NewtonInterpolation model with input data X and Y.
        """
        self.X = np.array(X, dtype=float)
        self.Y = np.array(Y, dtype=float)
        self.n = len(X)
        self.coefficients = np.zeros((self.n, self.n))  # Divided differences table
        self.x_symbol = symbols('x')  # Symbolic variable for polynomial
        self.title = title  # Title for the plot
        self.iterated_polynomials = []  # Store polynomial expressions iteratively

    def divided_difference_table(self):
        """
        Builds the divided difference table and extracts the coefficients.
        """
        self.coefficients[:, 0] = self.Y  # First column is the Y values

        # Calculate divided differences
        for j in range(1, self.n):
            for i in range(self.n - j):
                self.coefficients[i, j] = (
                    (self.coefficients[i + 1, j - 1] - self.coefficients[i, j - 1])
                    / (self.X[i + j] - self.X[i])
                )

        return self.coefficients  # Return the full table

    def construct_polynomial(self):
        """
        Construct the interpolated polynomial symbolically.
        """
        polynomial = 0  # Start with an empty polynomial
        term = 1  # Term multiplier

        # Build the polynomial using the divided difference coefficients
        for i in range(self.n):
            polynomial += self.coefficients[0, i] * term
            self.iterated_polynomials.append(simplify(polynomial))  # Store the current polynomial
            if i < self.n - 1:
                term *= (self.x_symbol - self.X[i])

        return simplify(polynomial)  # Return the final simplified polynomial

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

    def save_to_excel(self, filename):
        """
        Save the divided differences table and iterated polynomials to an Excel file.
        """
        # Convert divided difference table to DataFrame
        df_divided_diff = pd.DataFrame(
            self.coefficients,
            columns=[f"Order {i}" for i in range(self.n)]
        )
        
        # Create a DataFrame for iterated polynomials
        df_polynomials = pd.DataFrame({
            "Polynomial Iteration": [str(poly) for poly in self.iterated_polynomials]
        })

        # Save to Excel with two sheets
        with pd.ExcelWriter(filename) as writer:
            df_divided_diff.to_excel(writer, sheet_name='Divided Differences', index=False)
            df_polynomials.to_excel(writer, sheet_name='Iterated Polynomials', index=False)

        print(f"Saved results to {filename}")

    def plot(self, polynomial):
        """
        Plot the original data points and the interpolated polynomial.
        """
        # Create a lambdified function from the symbolic polynomial for numerical evaluation
        polynomial_func = lambdify(self.x_symbol, polynomial)

        X_interp = np.linspace(min(self.X), max(self.X), 100)
        Y_interp = polynomial_func(X_interp)

        plt.scatter(self.X, self.Y, color='red', label='Data Points')
        plt.plot(X_interp, Y_interp, color='blue', label='Interpolated Polynomial')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Newton Interpolation - {self.title}')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage with multiple datasets
datasets = [
    ([0.0, 1.0, 2.0, 3.0], [1.0, 2.7182, 7.3891, 20.0855], 'Dataset 1'),
    ([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [1.0, 2.1190, 2.9100, 3.9450, 5.7200, 8.6950], 'Dataset 2'),
    ([1.0, 2.0, 3.0, 5.0, 6.0], [4.75, 4.0, 5.25, 19.75, 36.0], 'Dataset 3'),
    ([0, 10, 20, 30, 40, 50, 60], [50000, 35000, 31000, 20000, 19000, 12000, 11000], 'Dataset 4')
]

# Process each dataset and save the results to Excel
for i, (X, Y, title) in enumerate(datasets):
    model = NewtonInterpolation(X, Y, title)
    model.divided_difference_table()

    # Construct and display the polynomial
    polynomial = model.construct_polynomial()
    print(f"Interpolated Polynomial for {title}: {polynomial}")

    # Plot the data and the polynomial
    model.plot(polynomial)

    # Save the results to Excel
    model.save_to_excel(f'newton_interpolation_{i+1}.xlsx')
