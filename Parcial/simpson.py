import numpy as np
from scipy.integrate import quad  # For exact integration

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
    print(y)
    print(x)
    print(h)

    # Apply Simpson's rule
    integral = (h / 3) * (y[0] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2]) + y[n])
    print(integral)
    return integral

def calculate_error(f, a, b, n):
    """
    Calculate the absolute error between the analytical integral and Simpson's rule.
    """
    # Exact integral using scipy's quad function
    I_real, _ = quad(f, a, b)  # _ captures the estimation error from quad
    I_simpson = simpson_rule(f, a, b, n)
    
    # Calculate the absolute error
    error = abs(I_real - I_simpson)
    
    return I_simpson, I_real, error

def f1_7(x):
    return 2-((x+2)/np.exp(x))

# Perform the integrations and print results
for i, (f, a, b, n) in enumerate([
    (f1_7, 0, 5, 10)
], 1):
    try:
        I_simpson, I_real, error = calculate_error(f, a, b, n)
        print(f"Integral 1.{i}: Simpson = {I_simpson:}, Real = {I_real:}, |E| = {error:}")
    except ValueError as e:
        print(f"Integral 1.{i}: {e}")
