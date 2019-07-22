import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

package_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(package_dir, "./Data/ex2data1.txt")
data = pd.read_csv(path, header=None, names=["Exam_1", "Exam_2", "Admitted"])
print(data.head())

# ------------------------------------------------------------------------------------------------
# We plot the initial data to get a grasp of the data set.
positive = data[data["Admitted"] == 1]
negative = data[data["Admitted"] == 0]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(
    positive["Exam_1"], positive["Exam_2"], s=50, c="g", marker="o", label="Admitted"
)
ax.scatter(
    negative["Exam_1"],
    negative["Exam_2"],
    s=50,
    c="r",
    marker="x",
    label="Not Admitted",
)
ax.legend()
ax.set_xlabel("Exam 1 Score")
ax.set_ylabel("Exam 2 Score")
# plt.show()

# ----------------------------------------------------------------------------------------------
# Logistic regression works with the sigmoid function, which for all its domain is
# contained between [0,1].


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# We are gong to define the cost function for the Logistic Regression.
# Unlike in the Linear regression code we did. We are going to do the
# conversion to numpy matrices in the function definition itself. This
# will allow us to have a cleaner code.


def cost_function(X, y, theta):
    # Define the numpy matrices for the calculations from the dataframes.
    theta = np.matrix(theta)
    X = np.matrix(np.array(X))
    y = np.matrix(np.array(y))
    # Define the two terms needed for the cost function.
    # Using the function np.multiply() we avoid needing to transpose certain vectors
    # for this code to work.
    inner = np.log(sigmoid(np.dot(X, theta.T)))
    first_term = np.multiply(-y.T, inner)
    second_term = np.multiply((1 - y).T, np.log(1 - sigmoid(np.dot(X, theta.T))))
    return np.sum(first_term - second_term) / len(X)


# Adding a column of 1's for the theta0 values.
data.insert(0, "Ones", 1)
# print(data.head())

# Defining the variables.
cols = data.shape[1]
X = data.iloc[:, 0 : cols - 1]
y = data.iloc[:, cols - 1 : cols]
theta = np.zeros(cols - 1)

print("Cost function test:", cost_function(X, y, theta))


# Now that we have a working cost function the next step is
# to define the gradient descent for the logistic regression.


# We don't actually loop here to reduce the value of the cost function.
# Since in the assignment in Andrews ML course we had to use fminuc to optimize
# the selection of parameters. We will reproduce this with scipy.
def gradient_implementation(X, y, theta):
    # Define the numpy matrices for the calculations from the dataframes.
    theta = np.matrix(theta)
    X = np.matrix(np.array(X))
    y = np.matrix(np.array(y))

    # Define number of parameters/features of our data set, (number of thetas).
    features = int(theta.shape[1])
    # Initialize the existence of the grad variable, which will be the output.
    grad = np.zeros(features)

    diff = sigmoid(np.dot(X, theta.T)) - y

    for i in range(features):
        term = np.multiply(diff, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad


# We will though try to write a function that mimics the results of scipy's library.
def true_gradient(X, y, theta, iterations, alpha):
    # Define the numpy matrices for the calculations from the dataframes.
    theta = np.matrix(theta)
    X = np.matrix(np.array(X))
    y = np.matrix(np.array(y))
    temp = np.matrix(np.zeros(theta.shape))

    # Define number of parameters/features of our data set, (number of thetas).
    features = int(theta.shape[1])
    # Initialize the existence of the grad variable, which will be the output.
    grad = np.zeros(features)
    # Initialize the existence of the cost variable.
    cost = np.zeros(iterations)

    for j in range(iterations):
        diff = sigmoid(np.dot(X, theta.T)) - y

        for i in range(features):
            term = np.multiply(diff, X[:, i])
            grad[i] = np.sum(term) / len(X)
            temp[0, i] = theta[0, i] - (alpha * grad[i])

        theta = temp
        cost[j] = cost_function(X, y, theta)

    return theta


g = true_gradient(X, y, theta, 1000, 0.01)
cost = cost_function(X, y, g)
print("Theta and cost", g, cost)

# --------------------------------------------------------------------------------------------------------------

# I correctly get the initial result for the cost function, but for some reason it does'nt stop spitting out errors
# I'm going to move on and come back in the future to fix these.
