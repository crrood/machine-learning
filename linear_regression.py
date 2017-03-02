# generate a random data set
# and run gradient descent on its cost function
# to determine weights for each of the inputs
#
# generalized to n parameters

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# create random independent data
M = 50

# takes a matrix of the form [[param1_mean, param1_range], ... [paramN_mean, paramN_range]]
def generate_data(characteristics):
	N = characteristics[:,0].size
	result = np.zeros(M * N).reshape(M, N)
	for i in range(N):
		result[:,i] = np.round(
			characteristics[i][0] - characteristics[i][1] / 2 + np.random.rand(M) * characteristics[i][1])
	return result

characteristics = np.array([
	[1500, 2000],
	[3, 4],
	[75, 150]])
X = np.concatenate([np.ones(M).reshape(M, 1), generate_data(characteristics)], axis=1)

# create dependent data
theta_true = np.array([3000, .6, 500, -10])
Y = np.matmul(X, theta_true)
Y = np.round(Y + np.random.randn(Y.size) * np.average(Y) / 10)

# scale data
X_scaled = preprocessing.scale(X, axis=0)
X_scaled[:,0] = np.ones(X_scaled[:,0].size)

print(M, "data points created based on", characteristics[:,0].size, "parameters")

# check the data
# print(X)
# print(Y)
# print(X_scaled)
#
# for i in range(1, X[0].size):
# 	plt.subplot(X[0].size - 1, 1, i)
# 	plt.plot(X[:,i], Y, 'ro')
# plt.show()

# define squared error function
def J(theta):
	return np.sum((np.matmul(X_scaled, theta) - Y) ** 2) / (2 * X_scaled[:,0].size)

# define derivative of squared error function
def dJ(theta):
	return np.sum(((np.matmul(X_scaled, theta) - Y)[:, np.newaxis] * X_scaled) / X_scaled[:,0].size, axis=0)

# variables for gradient descent
ALPHA = 0.001
epsilon_limits = np.array([0.1, 0.3, 1.])
SMALLEST_EPSILON_LIMIT = np.min(epsilon_limits)
epsilon = 100. # initialized to an arbitrary value greater than EPSILON_LIMIT
theta, theta_delta = np.zeros(X_scaled[0].size), np.zeros(X_scaled[0].size)
theta_results = np.zeros(epsilon_limits.size * theta.size).reshape(epsilon_limits.size, theta.size)

i = 0
MAX_ITERATIONS = 30000

# arrays to store data for analyzing gradient descent
j_history = np.zeros(MAX_ITERATIONS)
theta_history = np.zeros(MAX_ITERATIONS * theta.size).reshape(MAX_ITERATIONS, theta.size)

# algoritm for gradient descent
# i limit will stop loop in case of divergence
while epsilon > SMALLEST_EPSILON_LIMIT and i < MAX_ITERATIONS:
	theta_delta = ALPHA * dJ(theta)
	epsilon = np.sum(abs(theta_delta))
	
	# store theta values at different epsilons for comparison
	for j in range(epsilon_limits.size):
		if epsilon < epsilon_limits[j]:
			theta_results[j] = theta
			epsilon_limits[j] = 0

	theta -= theta_delta
	i += 1

	# store data for analyzing performance
	j_history[i] = J(theta)
	theta_history[i] = theta

# check result
print("------")
print("iterations: ", i)
print("last epsilon: ", epsilon)
print("------")
print("theta:", theta)
print("theta_results:\n", theta_results)

# check for convergence
# plt.plot(range(i), j_history[:i])
for j in range(theta_history[0].size):
	plt.plot(range(i), theta_history[:i,j])
plt.show()

# compute prediction errors
errors = np.matmul(X_scaled, theta) - Y

# print("------")
# print("errors: ", errors)
print("total error: ", np.sum(abs(errors)))

# plot regressions vs data
for i in range(1, X[0].size):
	plt.subplot(X[0].size - 1, 1, i)
	for j in range(theta_results[:,0].size):
		plt.plot([np.min(X[:,i]), np.max(X[:,i])], 
			[np.min(X_scaled[:,i]) * theta_results[j][i] + theta_results[j][0], 
			np.max(X_scaled[:,i]) * theta_results[j][i] + theta_results[j][0]])
	plt.plot(X[:,i], Y, 'r.')
plt.show()