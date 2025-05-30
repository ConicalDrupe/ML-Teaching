import numpy as np
import pandas as pd

def my_vectorized_gradient_descent(X,y,learning_rate=0.001,iterations=100):
    """
    X - Array of numbers
    * Hint calculate base error first
    """

    # add intercept
    X.insert(loc=0, column='intercept',value=np.ones(X.shape[0]))

    m = X.shape[0]
    assert m == len(y)

    loss_history = []

    # initialize params
    theta = np.zeros(X.shape[1])


    for i in range(iterations):

        # Calculate prediction
        prediction = X @ theta

        # Calculate Cost
        cost = (1/(2*m))* (np.sum((prediction-y)**2))

        # Calculate Gradients
        gradients = (1/m) * X.T @ (prediction-y)

        # updating parameter
        theta = theta - learning_rate*gradients

        # Append loss
        loss_history.append(cost)

        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{iterations}, Cost: {cost:.6f}")

    return theta,loss_history

if __name__ == "__main__":
    import pandas as pd
    import os


    data = os.path.join(os.getcwd(),'fake_housing_data.csv')
    df = pd.read_csv(data)

    print(df.columns)

    X = df[['lot_size_sqft','garage_spaces']]
    print(X.head())

    for col in X.columns:
        mu = np.mean(X[col])
        std = np.std(X[col])
        X[col] = X[col].map(lambda x: (x-mu)/std )

    y = df['price']

    iterations = 1000
    lr = 0.01
    theta, loss_history = my_vectorized_gradient_descent(X,y,learning_rate=lr,iterations=iterations)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot([i for i in range(iterations)],loss_history)
    plt.show()


    import seaborn as sns

    # def corr_plot(df):
    #     coor_mat = df.select_dtypes(include='number').corr()
    #     sns.heatmap(coor_mat, annot=True)
    #     plt.show()

    # corr_plot(df)

    print(theta)


    def plot_regression_surface(df, theta, feature_stats):
        """
        Plot the 3D surface of the fitted linear regression plane
        
        Parameters:
        df: DataFrame with original data
        theta: fitted parameters [intercept, lot_size_coef, garage_coef]
        feature_stats: dict with normalization statistics
        """
        
        # Get original feature ranges for plotting
        lot_size_min, lot_size_max = df['lot_size_sqft'].min(), df['lot_size_sqft'].max()
        garage_min, garage_max = df['garage_spaces'].min(), df['garage_spaces'].max()
        
        # Create meshgrid for surface plot (in original scale)
        lot_size_range = np.linspace(lot_size_min, lot_size_max, 50)
        garage_range = np.linspace(garage_min, garage_max, 20)
        lot_mesh, garage_mesh = np.meshgrid(lot_size_range, garage_range)
        
        # Normalize the meshgrid values (same way as training data)
        lot_mesh_norm = (lot_mesh - feature_stats['lot_size_mean']) / feature_stats['lot_size_std']
        garage_mesh_norm = (garage_mesh - feature_stats['garage_mean']) / feature_stats['garage_std']
        
        # Calculate predictions using the fitted plane equation
        # price = theta[0] + theta[1]*lot_size_norm + theta[2]*garage_norm
        price_mesh = theta[0] + theta[1] * lot_mesh_norm + theta[2] * garage_mesh_norm
        
        # Create the 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surface = ax.plot_surface(lot_mesh, garage_mesh, price_mesh, 
                                 alpha=0.7, cmap='viridis', 
                                 linewidth=0, antialiased=True)
        
        # Scatter plot of actual data points
        ax.scatter(df['lot_size_sqft'], df['garage_spaces'], df['price'], 
                  color='red', s=50, alpha=0.8, label='Actual Data')
        
        # Labels and title
        ax.set_xlabel('Lot Size (sq ft)', fontsize=12)
        ax.set_ylabel('Garage Spaces', fontsize=12)
        ax.set_zlabel('Price ($)', fontsize=12)
        ax.set_title('Linear Regression: Price vs Lot Size & Garage Spaces\n(Red dots = Actual data, Surface = Fitted plane)', 
                     fontsize=14, pad=20)
        
        # Add colorbar
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=20, label='Predicted Price ($)')
        
        # Add legend
        ax.legend(loc='upper left')
        
        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print the plane equation
        print(f"\nFitted Plane Equation (normalized features):")
        print(f"Price = {theta[0]:.2f} + {theta[1]:.2f} * lot_size_norm + {theta[2]:.2f} * garage_norm")

    feature_stats = {
        'lot_size_mean': df['lot_size_sqft'].mean(),
        'lot_size_std': df['lot_size_sqft'].std(), 
        'garage_mean': df['garage_spaces'].mean(),
        'garage_std': df['garage_spaces'].std()
    }

    # Plot the regression surface
    plot_regression_surface(df, theta, feature_stats)
