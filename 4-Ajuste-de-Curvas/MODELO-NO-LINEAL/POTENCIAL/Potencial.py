import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PowerRegression:
    def __init__(self, X, Y, title):
        """
        Initialize the PowerRegression model with input data X and Y.
        """
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.n = len(X)
        self.title = title  # Title for identification
        self.steps = []  # Store fit steps for Excel export
        self.intermediate_steps = []  # Store row-wise calculations for Excel export
        self.a = None  # Coefficient A (base of the power model)
        self.b = None  # Coefficient B (exponent)

    def log_step(self, description, value):
        """
        Log each step of the fit process for tracking.
        """
        self.steps.append({'Step': description, 'Value': value})

    def fit(self):
        """
        Fit the power model by linearizing the data and calculating coefficients.
        """
        # Linearization: Transform X to ln(X) and Y to ln(Y)
        log_X = np.log(self.X)
        log_Y = np.log(self.Y)

        # Store individual row-wise calculations (xi, yi, ln(xi), ln(yi), ln(xi) * ln(yi))
        for i, (x, y) in enumerate(zip(self.X, self.Y), start=1):
            ln_xi = np.log(x)
            ln_yi = np.log(y)
            ln_xi2 = ln_xi ** 2
            ln_xi_ln_yi = ln_xi * ln_yi
            self.intermediate_steps.append({
                'i': i, 'xi': x, 'yi': y, 'ln(xi)': ln_xi, 
                'ln(yi)': ln_yi, 'ln(xi)^2': ln_xi2, 'ln(xi) * ln(yi)': ln_xi_ln_yi
            })

        # Perform linear regression on (ln(X), ln(Y))
        sum_ln_x = np.sum(log_X)
        sum_ln_y = np.sum(log_Y)
        sum_ln_x_ln_y = np.sum(log_X * log_Y)
        sum_ln_x2 = np.sum(log_X ** 2)

        # Log all intermediate steps
        self.log_step("Sum of ln(X) (Σln(x))", sum_ln_x)
        self.log_step("Sum of ln(Y) (Σln(y))", sum_ln_y)
        self.log_step("Sum of ln(X) * ln(Y) (Σln(x) * ln(y))", sum_ln_x_ln_y)
        self.log_step("Sum of ln(X)^2 (Σln(x)^2)", sum_ln_x2)

        # Calculate coefficients B and ln(A)
        numerator_b = self.n * sum_ln_x_ln_y - sum_ln_x * sum_ln_y
        denominator_b = self.n * sum_ln_x2 - sum_ln_x ** 2
        self.b = numerator_b / denominator_b
        ln_a = (sum_ln_y - self.b * sum_ln_x) / self.n
        self.a = np.exp(ln_a)  # A = e^(ln(A))

        # Log the coefficients
        self.log_step("Numerator for B", numerator_b)
        self.log_step("Denominator for B", denominator_b)
        self.log_step("Coefficient B", self.b)
        self.log_step("Coefficient A (after exponentiation)", self.a)

    def predict(self, X):
        """
        Predict Y values for given X using the fitted power model.
        """
        return self.a * np.array(X) ** self.b

    def display_model(self):
        """
        Display the regression equation.
        """
        print(f"Power Regression Model: y = {self.a:.6f} * x^{self.b:.6f}")

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
        Plot the original data points and the fitted power curve.
        """
        X_range = np.linspace(min(self.X), max(self.X), 100)
        Y_pred = self.predict(X_range)

        plt.scatter(self.X, self.Y, color='blue', label='Data Points')
        plt.plot(X_range, Y_pred, color='red', label=f'y = {self.a:.2f} * x^{self.b:.2f}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Power Regression - {self.title}')
        plt.legend()
        plt.grid(True)
        plt.show()

"""
# Example usage with multiple datasets
datasets = [
    ([1, 2, 3, 4, 5, 6, 7], [0.5, 2.5, 2.0, 4.0, 3.5, 6.0, 5.5], 'Ejercicio 1.1'),
    ([1, 3, 5, 7, 10, 12, 13, 16, 18, 20], [3, 2, 6, 6, 8, 7, 10, 9, 12, 10], 'Ejercicio 1.2'),
    ([1, 2, 3, 4, 5], [0.5, 1.7, 3.4, 5.7, 8.4], 'Ejercicio 1.3'),
    ([1, 2, 2.5, 4, 6, 8, 8.5], [0.4, 0.7, 0.8, 1, 1.2, 1.3, 1.4], 'Ejercicio 1.4'),
    ([0.05, 0.4, 0.8, 1.2, 1.6, 2, 2.4], [550, 750, 1000, 1400, 2000, 2700, 3750], 'Ejercicio 1.5')
]

# Process each dataset and save the results to Excel
for i, (X, Y, title) in enumerate(datasets):
    model = PowerRegression(X, Y, title)
    model.fit()
    model.display_model()
    model.plot()
    model.save_to_csv(f'power_regression_steps_{i+1}.xlsx')
"""
