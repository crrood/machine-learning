import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# create semi-random data
sq_footage = np.arange(1000, 3000, 100)
sq_footage = np.round(sq_footage + np.random.randn(sq_footage.size) * 300)

num_bedrooms = np.arange(1, 3, .1)
num_bedrooms = np.round(num_bedrooms + np.random.randn(num_bedrooms.size))
num_bedrooms += (num_bedrooms < 1) + 0

X = np.array([np.ones(sq_footage.size), sq_footage, num_bedrooms]).T

# based around the line f(X) = 500 + 2X[1] + 1000X[2]
theta_true = np.array([500, 2, 1000])
Y = np.matmul(X, theta_true)
Y = np.round(Y + np.random.randn(Y.size) * np.average(Y) / 10)

# scale data
X_scaled = preprocessing.scale(X, axis=0)
X_scaled[:,0] = np.ones(X_scaled[:,0].size)

# check the data
# print(X)
# print(Y)
# print(X_scaled)
#
# for i in range(1, X[0,:].size):
# 	plt.subplot(X[0,:].size - 1, 1, i)
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
epsilon_limits = np.array([0.001, 0.1, 1.])
SMALLEST_EPSILON_LIMIT = np.min(epsilon_limits)
epsilon = 100. # initialized to an arbitrary value greater than EPSILON_LIMIT
theta, theta_delta = np.zeros(X_scaled[0,:].size), np.zeros(X_scaled[0,:].size)
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
# plt.plot(range(20), theta_history[:20,0], range(20), theta_history[:20,1])
# plt.show()

# compute prediction errors
errors = np.matmul(X_scaled, theta) - Y

# print("------")
# print("errors: ", errors)
print("total error: ", np.sum(abs(errors)))

# plot regressions vs data
for i in range(1, X[0].size):
	plt.subplot(X[0].size - 1, 1, i)
	for j in range(theta_results[:,0].size):
		plt.plot([np.min(X_scaled[:,i]), np.max(X_scaled[:,i])], 
			[np.min(np.matmul(X_scaled, theta_results[j])), 
			np.max(np.matmul(X_scaled, theta_results[j]))])
	plt.plot(X_scaled[:,i], Y, 'ro')
plt.show()