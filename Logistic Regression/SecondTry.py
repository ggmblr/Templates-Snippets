# Initialize libraries that are needed for the program.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load the file into a dataframe.
path = '/Users/simon/Desktop/DataScience/PythonProjects/Machine Learning/Logistic Regression/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam_1','Exam_2','Admitted'])
# Insert a column of ones neede for theta0.
data.insert(0,'Ones',1)
# print(data.head())


def sigmoid(z):
	return 1/(1 + np.exp(-z))


def cost_function(theta, x, y):
	m = len(x)
	z = x * theta
	h = sigmoid(z)
	J = (-y.T * np.log(h) - (1 - y).T * np.log(1 - h))/m
	return J

cols = data.shape[1]
train = np.matrix(np.array(data.iloc[:, 0:cols-1]))
y = np.matrix(np.array(data.iloc[:, cols-1:cols]))
theta_i = np.matrix(np.zeros(3)).T

# print(train, '\n', 'train',train.shape, '\n',y,'\n','y',y.shape,'\n','theta',theta_i, theta_i.shape)
print('Test cost function:', cost_function(theta_i,train,y))


def gradient(theta, x, y):
	m = len(y)
	z = x * theta
	h = sigmoid(z)
	temp = (x.T * (h - y))/m
	return temp


print('Test for gradient:', gradient(theta_i,train,y))



def g_descent(theta,x,y,alpha,iterations):
	temp = np.matrix(np.zeros(3)).T
	for i in range(iterations):
		temp = theta - alpha * gradient(temp,x,y)
	return temp

gdes = g_descent(theta_i,train,y,0.001,1000)
print('Test for gradient descent:', gdes)

test = np.matrix(np.array([1,45,85]))

print('Test for logistic regression:', sigmoid(test * gdes))

print('Test for cost after gradient descent:', cost_function(gdes,train,y),'\n')

#-----------------------------------------------------------------------------------------------------------------

theta_f = np.matrix(np.array([-24,0.2,0.2])).T

print('\n','Test for gradient:', gradient(theta_f,train,y))
print('Test cost function:', cost_function(theta_f,train,y))
gdes = g_descent(theta_f,train,y,0.001,100000)
print('\n','Test for gradient descent:', gdes)
print('Final cost function:', cost_function(gdes,train,y))

# Test for gradient: [[0.04290299]
# [2.56623412]
# [2.64679737]]
# Test cost function: [[0.21833019]]

# Test for gradient descent: [[-24.00002931]
# [  0.19826942]
# [  0.1981689 ]]
# Final cost function: [[0.21050428]]

# This is the output, which only differs slightly to the values desired in the assignment
# Expected cost (approx): 0.203
# theta:
#  -25.161343
#  0.206232
#  0.201472

#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
