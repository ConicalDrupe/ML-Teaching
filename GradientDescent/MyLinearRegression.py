class MyGradDescent:
    def __init__(self):
        pass

import numpy as np
import random
import matplotlib.pyplot as plt


class LR:
    def __init__(self,X,y,learning_rate,iters):
        "X - NxM pandas DataFrame"
        "y - vector of observations. nx1"
        self.X = X 
        self.y = y
        self.learning_rate = learning_rate
        self.iters = iters
        
        self._coef = None

    def fit(self,X,y,returnHistory=0,init_params=None):


        if not init_params:
            b = np.zeros(self.X.shape[0])
        else:
            b = init_params
        loss_history=[]

        n = X.shape[0] #else len(X) for numpy array
        assert n == len(self.y)

        for i in range(self.iters):

            error_sum = 0
            grad_sum = 0

            # Calculate prediction
            prediction = X @ b
            # Calculate errors
            avg_error = np.mean((prediction - y)**2) # using mse, vs using rmse
            rmse = (avg_error)**1/2

            # Calculate gradient
                # of ( X@b - y ) * ( X.T @ y)
            gradient =  2* ((prediction - y)**2) @ (X.T) 

        # Calculate average gradient, average loss
        avg_gradient = 1/n * grad_sum


            # updating parameter
            a = a - self.learning_rate*avg_gradient

            # Calculating Avg Loss
            avg_loss = 1/n * error_sum
            # Append loss
            if returnHistory:
                loss_history.append(avg_loss)
                if (i + 1) % 10 == 0:
                    print(f"Iteration {i+1}/{self.iters}, Cost: {avg_loss:.6f}")

        return loss_history







def my_grad_descent(X,y,learning_rate=0.001,iterations=100):
    """
    X - Array of numbers
    * Hint calculate base error first
    """
    n = len(X)
    assert len(X) == len(y)

    loss_history = []

    # initialize params
    a = 1

    for i in range(iterations):

        error_sum = 0
        grad_sum = 0
        for j in range(n):
            # Calculate prediction
            prediction = X[j]*a
            # Calculate errors
            error = (prediction - y[j])
            error_sum += error**2

            # Calculate gradient
            grad_sum += 2*error*X[j]

        # Calculate average gradient, average loss
        avg_gradient = 1/n * grad_sum


        # updating parameter
        a = a - learning_rate*avg_gradient

        # Calculating Avg Loss
        avg_loss = 1/n * error_sum
        # Append loss
        loss_history.append(avg_loss)

        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{iterations}, Cost: {avg_loss:.6f}")

    return a,loss_history

def main():
    # Set a random seed for reproducibility
    random.seed(42)
    
    # Generate some sample data
    m = 100  # number of examples
    X = []
    y = []
    
    for i in range(m):
        x_i = 2 * random.random()
        # True relationship: y = 4 + 3x + noise
        y_i = 4 + 3 * x_i + random.gauss(0, 1)
        X.append(x_i)
        y.append(y_i)

    iterations = 1000
    learning_rate = 0.01
    
    # Run gradient descent
    theta_0, cost_history = my_grad_descent( X, y, learning_rate=learning_rate, iterations=iterations)
    
    
    # Print the learned parameters
    print(f"\nLearned parameters: Slope = {theta_0:.4f}")
    
    # Plot the data and the regression line
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.subplot(2, 1, 1)
    plt.scatter(X, y)
    
    # Plot the regression line
    x_line = [0, 2]
    y_line = [0, theta_0 * 2]
    plt.plot(x_line, y_line, 'r-', linewidth=2, label='Predictions')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression with Gradient Descent')
    plt.legend()
    
    # Plot the cost history
    plt.subplot(2, 1, 2)
    plt.plot(range(1, iterations + 1), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost History')
    
    plt.tight_layout()
    plt.show()


def plot_lc():



def predict(x_train,y_true):
    pass
