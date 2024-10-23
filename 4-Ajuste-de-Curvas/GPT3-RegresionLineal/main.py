import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, X, Y):
        """
        Initialize the LinearRegression model with input data X and Y.
        """
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.n = len(X)
        self.a0 = 0  # Intercept
        self.a1 = 0  # Slope
    
    def fit(self):
        """
        Fit the linear regression model by calculating the coefficients.
        """
        sum_x = np.sum(self.X)
        sum_y = np.sum(self.Y)
        sum_xy = np.sum(self.X * self.Y)
        sum_x2 = np.sum(self.X ** 2)

        # Calculating slope (a1) and intercept (a0)
        self.a1 = (self.n * sum_xy - sum_x * sum_y) / (self.n * sum_x2 - sum_x ** 2)
        self.a0 = (sum_y - self.a1 * sum_x) / self.n

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

    def plot(self):
        """
        Plot the original data points and the regression line.
        """
        Y_pred = self.predict(self.X)

        # Plotting the data points and the regression line
        plt.scatter(self.X, self.Y, color='blue', label='Data Points')
        plt.plot(self.X, Y_pred, color='red', label=f'Regression Line: y = {self.a0:.6f} + {self.a1:.6f}x')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression - Least Squares Method')
        plt.legend()
        plt.grid(True)
        plt.show()

# Usage
X = [1, 2, 3, 4, 5, 6, 7]
Y = [0.5, 2.5, 2.0, 4.0, 3.5, 6.0, 5.5]

# Create an instance of the LinearRegression class
model = LinearRegression(X, Y)

# Fit the model (calculate coefficients)
model.fit()

# Display the regression equation
model.display_model()

# Plot the data and the regression line
model.plot()
