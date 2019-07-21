#from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

package_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(package_dir, './Data/ex1data1.txt')
#path = './Data/ex1data1.txt'
#path = '/Users/simon/Desktop/DataScience/PythonProjects/Machine Learning/Linear regression/Data/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
print(data.head())

#data.plot(kind='scatter', x='Population', y='Profit')
# plt.show()


# We define the cost function for linear regression.
def cost_function(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    # Given the dimensions of x,y,theta, when transforming them into np arrays,
    # x=(97x2), y(1x97) theta(1x2), we need to transpose Theta and y for this
    # vectorization to work.
    return np.sum(inner) / (2 * len(X))


# Adding a column of ones that is needed for the theta0 values.
data['ones'] = 1
# Sorting the columns of the df to make the Ones column the first one.
data = data[['ones', 'Population', 'Profit']]

# Defining the variables for our Linear Regression.
cols = data.shape[1]
Xdf = data.iloc[:, 0:cols-1]
ydf = data.iloc[:, cols-1:cols]
# Defining ydf this way gives us a mx1 vector once we change it into a numpy matrix,
# if we instead use ydf = data.iloc[:,2], we get a 1xm vector, thus needing to transpose
#  it when computing the cost and gradient functions. This way we have finally a completely
# general code for these functions.
# Not sure if I need X as a vector or a mx2 matrix->(I). WE DO!
Idf = data.iloc[:, [0, 1]]

# Defining Theta vector.
theta = np.matrix(np.array([0, 0]))
thetadf = pd.DataFrame(theta)

# I'm going to try to get the calculations done with dataframes and with numpy matrices.

X = np.matrix(Xdf)
y = np.matrix(ydf)
I = np.matrix(Idf)
print(X.shape, y.shape, I.shape, theta.shape)

print(cost_function(I, y, theta))


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
            temp[0, j] = theta[0, j] - (alpha/len(X)) * np.sum(term)

        theta = temp
        cost[i] = cost_function(X, y, theta)

    return theta, cost


# Define parameters for the gradient descent.
alpha = 0.01
iterations = 1000

g, cost = gradient(I, y, theta, alpha, iterations)
print(g)
print(cost_function(I, y, g))

# Notice that given the form we appended the columns of 'ones' to the X df, we now have to
# read g backwards, being the first term the dependent term of the line eq. and the second
# the independent term.
# WE FIXED THIS BY ORDERING THE COLUMNS OF THE DF CORRECTLY.
# The last thing to do is visualize the linear regression over the data.
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)


fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iterations), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


# -----------------------------------------------------------------------------------

# Instead of writing these functions ourselves we could have simply used
# the commonly used python library: Scikit-learn. Let's see how this would look like:


model = linear_model.LinearRegression()
fitt = model.fit(I, y)

x = np.array(X[:, 1].A1)
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Sklearn: Predicted Profit vs. Population Size')
plt.show()
