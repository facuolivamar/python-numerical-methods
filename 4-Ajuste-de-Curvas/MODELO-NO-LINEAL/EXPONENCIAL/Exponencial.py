import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ExponentialRegression:
    def __init__(self, X, Y, title):
        """
        Initialize the ExponentialRegression model with input data X and Y.
        """
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.n = len(X)
        self.title = title  # Title for identification
        self.steps = []  # Store fit steps for Excel export
        self.intermediate_steps = []  # Store row-wise calculations for Excel export
        self.a = None  # Coefficient A (base of exponential)
        self.b = None  # Coefficient B (exponent)

    def log_step(self, description, value):
        """
        Log each step of the fit process for tracking.
        """
        self.steps.append({'Step': description, 'Value': value})

    def fit(self):
        """
        Fit the exponential model by linearizing the data and calculating coefficients.
        """
        # Linearization: Transform Y to ln(Y)
        log_Y = np.log(self.Y)

        # Store individual row-wise calculations (xi, yi, ln(yi), xi^2, xi * ln(yi))
        for i, (x, y) in enumerate(zip(self.X, self.Y), start=1):
            ln_yi = np.log(y)
            xi2 = x ** 2
            xi_ln_yi = x * ln_yi
            self.intermediate_steps.append({
                'i': i, 'xi': x, 'yi': y, 'ln(yi)': ln_yi,
                'xi^2': xi2, 'xi * ln(yi)': xi_ln_yi
            })

        # Perform linear regression on (X, ln(Y))
        sum_x = np.sum(self.X)
        sum_log_y = np.sum(log_Y)
        sum_x_log_y = np.sum(self.X * log_Y)
        sum_x2 = np.sum(self.X ** 2)

        # Log all intermediate steps
        self.log_step("Sum of X (Σx)", sum_x)
        self.log_step("Sum of ln(Y) (Σln(y))", sum_log_y)
        self.log_step("Sum of X * ln(Y) (Σx * ln(y))", sum_x_log_y)
        self.log_step("Sum of X^2 (Σx²)", sum_x2)

        # Calculate coefficients B and ln(A)
        numerator_b = self.n * sum_x_log_y - sum_x * sum_log_y
        denominator_b = self.n * sum_x2 - sum_x ** 2
        self.b = numerator_b / denominator_b
        ln_a = (sum_log_y - self.b * sum_x) / self.n
        self.a = np.exp(ln_a)  # A = e^(ln(A))

        # Log the coefficients
        self.log_step("Numerator for B", numerator_b)
        self.log_step("Denominator for B", denominator_b)
        self.log_step("Coefficient B", self.b)
        self.log_step("Coefficient A (after exponentiation)", self.a)

    def predict(self, X):
        """
        Predict Y values for given X using the fitted exponential model.
        """
        return self.a * np.exp(self.b * np.array(X))

    def display_model(self):
        """
        Display the regression equation.
        """
        print(f"Exponential Regression Model: y = {self.a:.6f} * e^({self.b:.6f} * x)")

    def save_to_csv(self, filename):
        """
        Save all fit steps and intermediate calculations to an Excel file.
        """
        # Convert steps and intermediate calculations to DataFrames
        df_steps = pd.DataFrame(self.steps)
        df_intermediate = pd.DataFrame(self.intermediate_steps)

        # Save to Excel with multiple sheets
        with pd.ExcelWriter(filename) as writer:
            df_intermediate.to_excel(writer, sheet_name='Intermediate Steps', index=False)
            df_steps.to_excel(writer, sheet_name='Fit Steps', index=False)

        print(f"Saved fit steps and intermediate steps to {filename}")

    def plot(self):
        """
        Plot the original data points and the fitted exponential curve.
        """
        X_range = np.linspace(min(self.X), max(self.X), 100)
        Y_pred = self.predict(X_range)

        plt.scatter(self.X, self.Y, color='blue', label='Data Points')
        plt.plot(X_range, Y_pred, color='red', label=f'y = {self.a:.2f} * e^({self.b:.2f} * x)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Exponential Regression - {self.title}')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage with multiple datasets
datasets = [
    ([1, 2, 3, 4, 5, 6, 7], [0.5, 2.5, 2.0, 4.0, 3.5, 6.0, 5.50], 'Exponential Exercise 1'),
    ([1, 3, 5, 7, 10, 12, 13, 16, 18, 20], [3, 2, 6, 6, 8, 7, 10, 9, 12, 10], 'Exponential Exercise 2'),
    ([1, 2, 3, 4, 5], [0.5, 1.7, 3.4, 5.7, 8.4], 'Exponential Exercise 3')
]

# Process each dataset and save the results to Excel
for i, (X, Y, title) in enumerate(datasets):
    model = ExponentialRegression(X, Y, title)
    model.fit()
    model.display_model()
    model.plot()
    model.save_to_csv(f'exponential_regression_steps_{i+1}.xlsx')
