import numpy as np
import pandas
import matplotlib.pyplot as plt

# create semi-random data
# loosely based around the line f(x) = 500 + 3x
X = np.arange(1000, 3000, 100)
X = list(map(lambda x: round(x + np.random.randn() * 300), X))
X = np.array(X)

Y = list(map(lambda x: round(1000 * np.random.rand() + x * 3 + np.random.randn() * 600), X))
Y = np.array(Y)

print(X)
print(Y)

# check the data
#plt.plot(X, Y, 'ro')
#plt.show()

# normalize data
X_norm = np.ones(X.size)
X_range = np.max(X) - np.min(X)
for i in range(X.size):
	X_norm[i] = (X[i] - np.average(X)) / X_range

#print(X_norm)

# define error function
def J(theta0, theta1):
	result = 0
	for i in range(X_norm.size):
		result += ((theta0 + theta1 * X_norm[i]) - Y[i])**2
	return result / (2 * (X_norm.size + 1))

# define derivative of error function
def dJ(theta0, theta1, param_index):
	result = 0
	for i in range(X_norm.size):
		# the final term is kind of hackish
		# will only work for univariate regressions
		result += ((theta0 + theta1 * X_norm[i]) - Y[i]) * X_norm[i] ** param_index
	return result / (X_norm.size + 1)

# variables for gradient descent
ALPHA = 0.1
EPSILON_LIMIT = 0.0001
epsilon = 100. # initialized to an arbitrary value about EPSILON_LIMIT
theta0 = 0.
theta1 = 0.
Jarray = np.ones(3000) # array to store data to check for rate of convergence
i = 0

# algoritm for gradient descent
# i limit will stop loop in case of divergence
while epsilon > EPSILON_LIMIT and i < 3000:
	Jarray[i] = J(theta0, theta1)
	theta0_temp = theta0 - ALPHA * dJ(theta0, theta1, 0)
	theta1_temp = theta1 - ALPHA * dJ(theta0, theta1, 1)
	epsilon = abs(theta1 - theta1_temp)
	theta0 = theta0_temp
	theta1 = theta1_temp
	i += 1

# check result
print(i)
print(epsilon)
print("------")
print("theta0:", theta0)
print("theta1:", theta1)

# print progression of J to check for convergence
#plt.plot(range(len(Jarray)), Jarray)
#plt.show()

# compute margin of error
errors = np.ones(X_norm.size)
total_error = 0
for i in range(X_norm.size - 1):
	errors[i] = (theta0 + theta1 * X_norm[i]) - Y[i]
	total_error += abs(errors[i])

print("------")
print("errors: ", errors)
print("total error: ", total_error)

# plot regression vs data
plt.plot([np.min(X_norm), np.max(X_norm)], [theta0 + theta1 * np.min(X_norm), theta0 + theta1 * X_norm[X_norm.size - 1]], 
	X_norm, Y, 'ro')
plt.show()