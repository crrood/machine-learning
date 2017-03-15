# generate a random data set
# and run gradient descent on its cost function
# to determine weights for each of the inputs
#
# generalized to n parameters

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# create random independent data
# takes a matrix of the form [[param1_mean, param1_range], ... [paramN_mean, paramN_range]]
M = 50
def generate_data(characteristics):
	result = np.zeros(M * N).reshape(M, N)
	for i in range(N):
		result[:,i] = np.round(
			characteristics[i][0] - characteristics[i][1] / 2 + np.random.rand(M) * characteristics[i][1])
	return result

characteristics = np.array([
	[1500, 2000],
	[3, 4],
	[75, 150]])
N = characteristics[:,0].size
X_true = np.concatenate([np.ones(M).reshape(M, 1), generate_data(characteristics)], axis=1)
N += 1 # account for added parameter of all ones

# create dependent data
theta_true = np.array([3000, .6, 500, -10])
Y_true = np.matmul(X_true, theta_true)
Y_true = np.round(Y_true + np.random.randn(M) * np.average(Y_true) / 10)

# scale data
X_scaled = preprocessing.scale(X_true, axis=0)
X_scaled[:,0] = np.ones(X_scaled[:,0].size)

print(M, "data points created based on", characteristics[:,0].size, "parameters")
print("------")

# check the data
# print(X_true)
# print(Y_true)
# print(X_scaled)
# print("------")
# 
# for i in range(1, X_true[0].size):
# 	plt.subplot(X_true[0].size - 1, 1, i)
# 	plt.plot(X_true[:,i], Y_true, 'ro')
# plt.show()

# define squared error function
def J(theta):
	return np.sum((np.matmul(X_scaled, theta) - Y_true) ** 2) / (2 * M)

# define derivative of squared error function
def dJ(theta):
	return np.sum(((np.matmul(X_scaled, theta) - Y_true)[:, np.newaxis] * X_scaled) / M, axis=0)

# variables for gradient descent
ALPHA = 0.001
epsilon_limits = np.array([0.01, 0.1, 0.3])
SMALLEST_EPSILON_LIMIT = np.min(epsilon_limits)
epsilon = 100. # initialized to an arbitrary value greater than EPSILON_LIMIT
theta_scaled, theta_delta = np.zeros(N), np.zeros(N)
theta_scaled_results = np.zeros(epsilon_limits.size * N).reshape(epsilon_limits.size, N)

i = 0
MAX_ITERATIONS = 30000

# arrays to store data for analyzing gradient descent
j_history = np.zeros(MAX_ITERATIONS)
theta_history = np.zeros(MAX_ITERATIONS * N).reshape(MAX_ITERATIONS, N)

# algoritm for gradient descent
# i limit will stop loop in case of divergence
while epsilon > SMALLEST_EPSILON_LIMIT and i < MAX_ITERATIONS:
	theta_delta = ALPHA * dJ(theta_scaled)
	epsilon = np.sum(abs(theta_delta))
	
	# store theta values at different epsilons for comparison
	for j in range(epsilon_limits.size):
		if epsilon < epsilon_limits[j]:
			theta_scaled_results[j] = theta_scaled
			epsilon_limits[j] = 0

	theta_scaled -= theta_delta
	i += 1

	# store data for analyzing performance
	j_history[i] = J(theta_scaled)
	theta_history[i] = theta_scaled

# check result
print("iterations: ", i)
print("last epsilon: ", epsilon)
print("theta_scaled:", theta_scaled)
#print("theta_scaled_results:\n", theta_scaled_results)
print("------")

# check for convergence
# plt.plot(range(i), j_history[:i])
for j in range(theta_history[0].size):
	plt.plot(range(i), theta_history[:i,j])
	plt.title("Theta history")
plt.show()

# compute fit to data
Y_calculated = np.matmul(X_scaled, theta_scaled)
R_squared = np.sum(np.square(Y_calculated - Y_true)) / np.sum(np.square(Y_true - np.average(Y_true)))

Y_composite = np.concatenate([Y_true.reshape(M, 1), Y_calculated.reshape(M, 1)], axis=1)
sort_order = np.argsort(Y_composite, axis=0)
Y_composite_sorted = np.zeros(M * 2).reshape(M, 2)
for i in range(M):
	Y_composite_sorted[i] = Y_composite[sort_order[i][0]]

print("R_squared: ", R_squared)
print("------")

plt.plot(range(M), Y_composite_sorted, 'o')
plt.show()

# plot regressions vs individual parameters
for i in range(1, N):
	plt.subplot(N - 1, 1, i)
	for j in range(theta_scaled_results[:,0].size):
		plt.plot([np.min(X_true[:,i]), np.max(X_true[:,i])], 
			[np.min(X_scaled[:,i]) * theta_scaled_results[j][i] + theta_scaled_results[j][0], 
			np.max(X_scaled[:,i]) * theta_scaled_results[j][i] + theta_scaled_results[j][0]])
	plt.plot(X_true[:,i], Y_true, 'r.')
plt.show()

# find un-scaled theta values to check against known characteristics
#
# since X_scaled * theta_scaled = X_true * theta_calculated = Y_calculated
# theta_calculated = X_true_inverse * X_scaled * theta_scaled

# first reshape matrices to be square so you can find their inverse
# and find average X values to avoid skewing the results too much
# M = number of data points
# N = number of paramters (including all-ones first parameter)
X_true_square = np.zeros(N ** 2).reshape(N, N)
X_scaled_square = np.zeros(N ** 2).reshape(N, N)
points_per_group = int(M / N)
for i in range(N):
	X_true_square[i] = np.average(X_true[i * points_per_group : (i + 1) * points_per_group], axis=0)
	X_scaled_square[i] = np.average(X_scaled[i * points_per_group : (i + 1) * points_per_group], axis=0)
X_true_square_inverse = np.linalg.inv(X_true_square)

theta_calculated = np.matmul(np.matmul(X_true_square_inverse, X_scaled_square), theta_scaled_results.T.reshape(N, theta_scaled_results.T.shape[1]))

print("theta_calculated.T: \n", theta_calculated.T)
print("theta_true: \n", theta_true)
print("------")

plt.title("% Error")
plt.plot(range(N), np.zeros(N), 'b--')
for i in range(theta_calculated.shape[1]):
	plt.plot(range(N), (theta_calculated[:,i] - theta_true) / theta_true, 'o')
plt.show()