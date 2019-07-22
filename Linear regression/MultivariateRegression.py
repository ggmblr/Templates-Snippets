import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

package_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(package_dir, './Data/ex1data2.txt')
#path = '/Users/simon/Desktop/DataScience/PythonProjects/Machine Learning/Linear regression/Data/ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

print(data.head())

# When observing the data we see that there are features with very different orders
# of magnitude. Thus applying feature scaling will help our code to converge faster.

data = (data - data.mean()) / data.std()
# As we saw in the Linear Regression if we simply append the column of ones, in the end result
# we are going to re-arrange the values of theta to correctly display the fitted model.
# Thus we are going to re-arrange the df to correctly match this now.
# We do this after the feature normalizatio, since the standard deviation for the Ones is 0,
# thus resulting in NaN's.
data['Ones'] = 1
data = data[['Ones', 'Size', 'Bedrooms', 'Price']]
print(data.head())

# Since we did the gradient descent with vectorized operations we can reuse the code
# from the linear regression code, and it will work with the multivariate one.


# We define the cost function for linear regression.
def cost_function(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    # Given the dimensions of x,y,theta, when transforming them into np arrays,
    # x=(97x2), y(1x97) theta(1x2), we need to transpose Theta and y for this
    # vectorization to work.
    return np.sum(inner) / (2 * len(X))


# Time to define the gradient descent function.
def gradient(X, y, theta, alpha, iterations):
    # Initialize the existence of the theta matrix for computations.
    temp = np.matrix(np.zeros(theta.shape))
    # Define number of parameters/features of our data set, (number of thetas).
    features = int(theta.shape[1])
    # Initialize the existence of the cost variable.
    cost = np.zeros(iterations)

    # Initialize loop over the set number of iterations.
    # Compute the difference between X*theta and y.
    for i in range(iterations):
        diff = (X * theta.T) - y

        # Initialize loop over number of features. And compute the new values for theta.
        for j in range(features):
            term = np.multiply(diff, X[:, j])
            temp[0, j] = theta[0, j] - (alpha / len(X)) * np.sum(term)

        theta = temp
        cost[i] = cost_function(X, y, theta)

    return theta, cost


cols = data.shape[1]
X2 = data.iloc[:, 0:cols - 1]
y2 = data.iloc[:, cols - 1:cols]

X2 = np.matrix(X2)
y2 = np.matrix(y2)
theta = np.matrix(np.array([0, 0, 0]))

alpha = 0.01
iterations = 1000

g, cost = gradient(X2, y2, theta, alpha, iterations)

print(cost_function(X2, y2, g))
print(g, cost)

# The following plot shows us how the difference between our fitted model and the
# real values for y is decreasing as we go further down the iterations.
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iterations), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
