import numpy as np

def simpson_rule(f, a, b, n):
    """
    Compute the integral of a function f over [a, b] using Simpson's rule with n sub-intervals.
    n must be even.
    """
    if n % 2 == 1:
        raise ValueError("Simpson's rule requires an even number of sub-intervals.")

    h = (b - a) / n  # Step size
    x = np.linspace(a, b, n + 1)  # n + 1 points (including both endpoints)
    y = f(x)  # Function values at the points

    # Apply Simpson's rule
    integral = (h / 3) * (y[0] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2]) + y[n])
    return integral

# Define the functions to be integrated
def f1_1(x):
    return 8 + 5 * np.cos(x)

def f1_2(x):
    return 1 - x - 4 * x**3 + 3 * x**5

def f1_3(x):
    return np.sin(5 * x + 1)

def f1_4(x):
    return x * np.exp(2 * x)

# Perform the integrations using Simpson's rule
print("1.1:", simpson_rule(f1_1, 0, np.pi, 10))
print("1.2:", simpson_rule(f1_2, -3, 5, 10))
print("1.3:", simpson_rule(f1_3, 0, 3 * np.pi / 20, 8))
print("1.4:", simpson_rule(f1_4, 0, 4, 10))

# Simpson’s rule for data points (assumes uniform spacing)
def simpson_from_points(X, Y):
    """
    Compute the integral using Simpson's rule for discrete points.
    Assumes that X is evenly spaced and contains an even number of intervals.
    """
    n = len(X) - 1  # Number of sub-intervals
    if n % 2 == 1:
        raise ValueError("Simpson's rule requires an even number of sub-intervals.")
    
    h = X[1] - X[0]  # Assuming uniform spacing
    integral = (h / 3) * (Y[0] + 4 * np.sum(Y[1:n:2]) + 2 * np.sum(Y[2:n-1:2]) + Y[n])
    return integral

try:
    # Data for 1.5
    X_1_5 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    Y_1_5 = np.array([1.0, 7.0, 4.0, 3.0, 5.0, 9.0])
    print("1.5:", simpson_from_points(X_1_5, Y_1_5))
except ValueError as e:
    print("1.5:", e)

try:
    # Data for 1.6
    X_1_6 = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    Y_1_6 = np.array([1.0, -4.0, -5.0, 2.0, 4.0, 9.0, 6.0, -3.0])
    print("1.6:", simpson_from_points(X_1_6, Y_1_6))
except ValueError as e:
    print("1.6:", e)

