import numpy as np
import pandas as pd

class EulersMethod:
    def __init__(self, f, f_prime, y0, t0, t_end, h):
        """
        Initialize the Euler's Method with function, derivative, initial value, interval, and step size.
        """
        self.f = f  # The function to integrate
        self.f_prime = f_prime  # Partial derivative of the function
        self.y0 = y0  # Initial value y(0)
        self.t0 = t0  # Start of the interval
        self.t_end = t_end  # End of the interval
        self.h = h  # Step size
        self.results_table = []  # Initialize results table to store output
        self.t_values, self.y_values = self.eulers_method()  # Compute values

    def eulers_method(self):
        """
        Compute the values using Euler's Method.
        """
        n = int((self.t_end - self.t0) / self.h)  # Number of steps
        t_values = np.linspace(self.t0, self.t_end, n + 1)  # Time points
        y_values = np.zeros(n + 1)  # Array to store y values
        y_values[0] = self.y0  # Initial condition

        for i in range(1, n + 1):
            y_prev = y_values[i - 1]
            t_prev = t_values[i - 1]

            # Euler’s method step: y_i+1 = y_i + h * f(t_i, y_i)
            y_values[i] = y_prev + self.h * self.f(t_prev, y_prev)

            # Calculate the local truncation error: E = (f'(x_i, y_i) / 2) * h^2
            error = self.calculate_error(t_prev, y_prev)

            # Log the result
            self.results_table.append({
                'i': i,
                'Xi': round(t_prev, 4),
                'Yi+1': round(y_values[i], 6),
                '|E|': round(error, 6)
            })

        return t_values, y_values

    def calculate_error(self, t, y):
        """
        Calculate the local truncation error: E = (f'(x_i, y_i) / 2) * h^2
        """
        return abs(self.f_prime(t, y) / 2) * self.h ** 2

    def save_to_excel(self, filename):
        """
        Save the results to an Excel file with the specified structure.
        """
        # Convert results to a DataFrame
        df_results = pd.DataFrame(self.results_table)

        # Save to Excel with a single sheet
        with pd.ExcelWriter(filename) as writer:
            df_results.to_excel(writer, sheet_name='Euler Results', index=False)

        print(f"Saved Euler's Method results to {filename}")

    def display_results(self):
        """
        Display the results of the Euler method.
        """
        print(f"\n{self.t0} - {self.t_end} with h={self.h} - Results:")
        print(pd.DataFrame(self.results_table))

# Define the function dy/dx = e^(0.8x) - 0.5y
def differential_equation(x, y):
    return (1/x)*(y**2+y)

# Define the partial derivative of f: f'(x, y) = d/dx [e^(0.8x) - 0.5y]
def partial_derivative(x, y):
    return -((y**2+y)/x**2)  # Only the partial derivative with respect to x

# Initialize parameters for the Euler method
y0, t0, t_end, h = -2, 1, 3, 0.2

# Create an instance of the Euler's Method class
euler_example = EulersMethod(differential_equation, partial_derivative, y0, t0, t_end, h)

# Display the results
euler_example.display_results()

# Save the results to Excel
euler_example.save_to_excel('euler_method_results.xlsx')
