import numpy as np
from scipy.integrate import quad  # For exact integration

def trapezoidal_rule(f, a, b, n):
    """
    Compute the integral of a function f over [a, b] using the trapezoidal rule with n sub-intervals.
    """
    h = (b - a) / n  # Step size
    x = np.linspace(a, b, n + 1)  # n + 1 points (including both endpoints)
    y = f(x)  # Function values at the points

    # Apply the trapezoidal rule
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral

def calculate_error(f, a, b, n):
    """
    Calculate the absolute error between the analytical integral and trapezoidal rule.
    """
    # Exact integral using scipy's quad function
    I_real, _ = quad(f, a, b)  # _ captures the estimation error from quad
    I_trap = trapezoidal_rule(f, a, b, n)
    
    # Calculate the absolute error
    error = abs(I_real - I_trap)
    
    return I_trap, I_real, error

# Define the functions to be integrated
def f1_1(x):
    return 8 + 5 * np.cos(x)

def f1_2(x):
    return 1 - x - 4 * x**3 + 3 * x**5

def f1_3(x):
    return np.sin(5 * x + 1)

def f1_4(x):
    return x * np.exp(2 * x)

# Perform the integrations and print results
for i, (f, a, b, n) in enumerate([
    (f1_1, 0, np.pi, 10),
    (f1_2, -3, 5, 10),
    (f1_3, 0, 3 * np.pi / 20, 8),
    (f1_4, 0, 4, 10)
], 1):
    I_trap, I_real, error = calculate_error(f, a, b, n)
    print(f"Integral 1.{i}: Trapezoidal = {I_trap:}, Real = {I_real:}, |E| = {error:}")

# Example datasets for numerical integration with given points
def trapezoidal_from_points(X, Y):
    """
    Compute the integral using the trapezoidal rule given discrete points.
    """
    n = len(X) - 1
    h = X[1] - X[0]  # Assuming uniform spacing
    integral = (h / 2) * (Y[0] + 2 * np.sum(Y[1:-1]) + Y[-1])
    return integral

# Data for 1.5
X_1_5 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
Y_1_5 = np.array([1.0, 7.0, 4.0, 3.0, 5.0, 9.0])
I_trap_1_5 = trapezoidal_from_points(X_1_5, Y_1_5)
print(f"1.5: Trapezoidal = {I_trap_1_5:.6f}")

# Data for 1.6
X_1_6 = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
Y_1_6 = np.array([1.0, -4.0, -5.0, 2.0, 4.0, 9.0, 6.0, -3.0])
I_trap_1_6 = trapezoidal_from_points(X_1_6, Y_1_6)
print(f"1.6: Trapezoidal = {I_trap_1_6:.6f}")
