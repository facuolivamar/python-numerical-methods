import numpy as np

def three_point_derivative(X, Y, h):
    """
    Compute the derivative using the three-point formula.
    """
    n = len(X)
    derivatives = np.zeros(n)

    # Forward difference for the first point
    derivatives[0] = (-3 * Y[0] + 4 * Y[1] - Y[2]) / (2 * h)

    # Centered difference for the interior points
    for i in range(1, n - 1):
        derivatives[i] = (Y[i + 1] - Y[i - 1]) / (2 * h)

    # Backward difference for the last point
    derivatives[n - 1] = (Y[n - 3] - 4 * Y[n - 2] + 3 * Y[n - 1]) / (2 * h)

    return derivatives

def five_point_derivative(X, Y, h):
    """
    Compute the derivative using the five-point formula.
    """
    n = len(X)
    derivatives = np.zeros(n)

    # Use centered differences where possible
    for i in range(2, n - 2):
        derivatives[i] = (-Y[i + 2] + 8 * Y[i + 1] - 8 * Y[i - 1] + Y[i - 2]) / (12 * h)

    # Use three-point forward or backward formulas at boundaries
    derivatives[0] = (-3 * Y[0] + 4 * Y[1] - Y[2]) / (2 * h)
    derivatives[1] = (Y[2] - Y[0]) / (2 * h)
    derivatives[n - 2] = (Y[n - 1] - Y[n - 3]) / (2 * h)
    derivatives[n - 1] = (3 * Y[n - 1] - 4 * Y[n - 2] + Y[n - 3]) / (2 * h)

    return derivatives

# Example usage
datasets = [
    ([0, 0.5, 1.0, 1.5, 2.0], [0, 0.4207, 0.4546, 0.0706, -0.3784], 0.5),  # Dataset 1.4
    ([0, 1, 2, 3, 4], [1, 1.1052, 1.2214, 1.3499, 1.4918], 1.0)  # Dataset 1.5
]

for i, (X, Y, h) in enumerate(datasets):
    print(f"\nDataset {i + 1}:")
    three_point_result = three_point_derivative(X, Y, h)
    five_point_result = five_point_derivative(X, Y, h)
    
    print("Three-Point Derivative:")
    print(three_point_result)

    print("Five-Point Derivative:")
    print(five_point_result)


# Define the functions for 1.1, 1.2, 1.3
def f1_1(x):
    return np.exp(x) * np.cos(x)

def f1_2(x):
    return 5 * np.exp(3 * x) * np.sin(2 * x)

def f1_3(x):
    return (x**3 + 3 * x - 2) / (x**2 - 4)

# Three-point and five-point derivative formulas
def three_point_derivative(f, x, h):
    n = len(x)
    derivatives = np.zeros(n)

    # Forward difference for the first point
    derivatives[0] = (-3 * f(x[0]) + 4 * f(x[1]) - f(x[2])) / (2 * h)

    # Centered difference for interior points
    for i in range(1, n - 1):
        derivatives[i] = (f(x[i + 1]) - f(x[i - 1])) / (2 * h)

    # Backward difference for the last point
    derivatives[n - 1] = (f(x[n - 3]) - 4 * f(x[n - 2]) + 3 * f(x[n - 1])) / (2 * h)

    return derivatives

def five_point_derivative(f, x, h):
    n = len(x)
    derivatives = np.zeros(n)

    # Use centered differences for interior points
    for i in range(2, n - 2):
        derivatives[i] = (-f(x[i + 2]) + 8 * f(x[i + 1]) - 8 * f(x[i - 1]) + f(x[i - 2])) / (12 * h)

    # Forward and backward difference for boundary points
    derivatives[0] = (-3 * f(x[0]) + 4 * f(x[1]) - f(x[2])) / (2 * h)
    derivatives[1] = (f(x[2]) - f(x[0])) / (2 * h)
    derivatives[n - 2] = (f(x[n - 1]) - f(x[n - 3])) / (2 * h)
    derivatives[n - 1] = (3 * f(x[n - 1]) - 4 * f(x[n - 2]) + f(x[n - 3])) / (2 * h)

    return derivatives

# Define intervals and step sizes for each function
intervals = [
    (0, 0.7, f1_1),  # 1.1
    (1, 2, f1_2),  # 1.2
    (-1, 1, f1_3)  # 1.3
]

# Perform differentiation and display results
for i, (start, end, func) in enumerate(intervals, 1):
    x_values = np.arange(start, end + 0.1, 0.1)  # Generate x values with step size 0.1
    three_point = three_point_derivative(func, x_values, 0.1)
    five_point = five_point_derivative(func, x_values, 0.1)

    print(f"\nExercise 1.{i}:")
    print("Three-Point Derivative:")
    print(three_point)
    print("Five-Point Derivative:")
    print(five_point)