import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:
    def __init__(self, X, Y, title):
        """
        Initialize the LinearRegression model with input data X and Y.
        """
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.n = len(X)
        self.a0 = 0  # Intercept
        self.a1 = 0  # Slope
        self.title = title  # Title for identification
        self.steps = []  # Store fit steps for CSV export
        self.intermediate_steps = []  # Store row-wise calculations for CSV export

    def log_step(self, description, value):
        """
        Log each step of the fit process for tracking.
        """
        self.steps.append({'Step': description, 'Value': value})

    def fit(self):
        """
        Fit the linear regression model by calculating the coefficients.
        """
        # Calculating individual row-wise values and storing them
        for i, (x, y) in enumerate(zip(self.X, self.Y), start=1):
            xi2 = x ** 2
            xiyi = x * y
            self.intermediate_steps.append({
                'i': i, 'xi': x, 'yi': y, 'xi^2': xi2, 'xiyi': xiyi
            })

        # Aggregate sums and log them
        sum_x = np.sum(self.X)
        sum_y = np.sum(self.Y)
        sum_xy = np.sum(self.X * self.Y)
        sum_x2 = np.sum(self.X ** 2)

        self.log_step("Sum of X (Σx)", sum_x)
        self.log_step("Sum of Y (Σy)", sum_y)
        self.log_step("Sum of XY (Σxy)", sum_xy)
        self.log_step("Sum of X^2 (Σx²)", sum_x2)

        # Calculating slope (a1) and intercept (a0)
        numerator_a1 = self.n * sum_xy - sum_x * sum_y
        denominator_a1 = self.n * sum_x2 - sum_x ** 2
        self.a1 = numerator_a1 / denominator_a1
        self.a0 = (sum_y - self.a1 * sum_x) / self.n

        self.log_step("Numerator for a1", numerator_a1)
        self.log_step("Denominator for a1", denominator_a1)
        self.log_step("Slope (a1)", self.a1)
        self.log_step("Intercept (a0)", self.a0)

    def predict(self, X):
        """
        Predict Y values for given X using the regression line.
        """
        return self.a0 + self.a1 * np.array(X)

    def display_model(self):
        """
        Display the regression equation.
        """
        print(f"Linear Regression Model: y = {self.a0:.6f} + {self.a1:.6f}x")

    def save_to_csv(self, filename):
        """
        Save all fit steps and intermediate calculations to a CSV file.
        """
        # Convert steps and intermediate calculations to DataFrames
        df_steps = pd.DataFrame(self.steps)
        df_intermediate = pd.DataFrame(self.intermediate_steps)

        # Concatenate both DataFrames
        with pd.ExcelWriter(filename) as writer:
            df_intermediate.to_excel(writer, sheet_name='Intermediate Steps', index=False)
            df_steps.to_excel(writer, sheet_name='Fit Steps', index=False)

        print(f"Saved fit steps and intermediate steps to {filename}")

    def plot(self):
        """
        Plot the original data points and the regression line.
        """
        Y_pred = self.predict(self.X)

        plt.scatter(self.X, self.Y, color='blue', label='Data Points')
        plt.plot(self.X, Y_pred, color='red', label=f'Regression Line: y = {self.a0:.6f} + {self.a1:.6f}x')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Linear Regression - {self.title}')
        plt.legend()
        plt.grid(True)
        plt.show()

"""
# List of datasets
datasets = [
    ([1, 2, 3, 4, 5, 6, 7], [0.5, 2.5, 2.0, 4.0, 3.5, 6.0, 5.5], 'Ejercicio 1.1'),
    ([1, 3, 5, 7, 10, 12, 13, 16, 18, 20], [3, 2, 6, 6, 8, 7, 10, 9, 12, 10], 'Ejercicio 1.2'),
    ([1, 2, 3, 4, 5], [0.5, 1.7, 3.4, 5.7, 8.4], 'Ejercicio 1.3'),
    ([1, 2, 2.5, 4, 6, 8, 8.5], [0.4, 0.7, 0.8, 1, 1.2, 1.3, 1.4], 'Ejercicio 1.4'),
    ([0.05, 0.4, 0.8, 1.2, 1.6, 2, 2.4], [550, 750, 1000, 1400, 2000, 2700, 3750], 'Ejercicio 1.5')
]

# Process each dataset and save the results to CSV
for i, (X, Y, title) in enumerate(datasets):
    model = LinearRegression(X, Y, title)
    model.fit()
    model.display_model()
    model.plot()
    model.save_to_csv(f'regression_steps_{i+1}.xlsx')
"""