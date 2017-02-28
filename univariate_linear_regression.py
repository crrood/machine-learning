import numpy as np
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
# plt.plot(X, Y, 'ro')
# plt.show()

# normalize data
X_norm = np.ones(X.size)
X_range = np.max(X) - np.min(X)
for i in range(X.size):
	X_norm[i] = (X[i] - np.average(X)) / X_range

# print(X_norm)

# define error function
# squared error
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
epsilon_limits = np.array([0.1, 1., 10.])
SMALLEST_EPSILON_LIMIT = np.min(epsilon_limits)
epsilon = 100. # initialized to an arbitrary value greater than EPSILON_LIMIT
theta, theta_temp = np.zeros(2), np.zeros(2)
theta_results = np.ones(epsilon_limits.size * theta.size).reshape(theta.size, epsilon_limits.size)
Jarray = np.ones(3000) # array to store data to check for rate of convergence
i = 0

# algoritm for gradient descent
# i limit will stop loop in case of divergence
while epsilon > SMALLEST_EPSILON_LIMIT and i < 3000:
	Jarray[i] = J(theta[0], theta[1])
	theta_temp[0] = theta[0] - ALPHA * dJ(theta[0], theta[1], 0)
	theta_temp[1] = theta[1] - ALPHA * dJ(theta[0], theta[1], 1)
	epsilon = abs(theta[1] - theta_temp[1]) + abs(theta[0] - theta_temp[0])
	
	# store theta values at different epsilons for comparison
	for j in range(epsilon_limits.size):
		if epsilon < epsilon_limits[j]:
			theta_results[0][j] = theta[0]
			theta_results[1][j] = theta[1]
			epsilon_limits[j] = 0

	theta[0] = theta_temp[0]
	theta[1] = theta_temp[1]
	i += 1

# check result
print(i)
print(epsilon)
print("------")
print("theta[0]:", theta[0])
print("theta[1]:", theta[1])

# print progression of J to check for convergence
# plt.plot(range(i), Jarray[:i])
# plt.show()

# compute margin of error
errors = np.ones(X_norm.size)
total_error = 0
for i in range(X_norm.size):
	errors[i] = (theta[0] + theta[1] * X_norm[i]) - Y[i]
	total_error += abs(errors[i])

print("------")
print("errors: ", errors)
print("total error: ", total_error)

# plot regressions vs data
for i in range(theta_results[0].size):
	plt.plot([np.min(X_norm), np.max(X_norm)], 
		[theta_results[0][i] + theta_results[1][i] * np.min(X_norm), 
		theta_results[0][i] + theta_results[1][i] * np.max(X_norm)])
plt.plot(X_norm, Y, 'ro')
plt.show()