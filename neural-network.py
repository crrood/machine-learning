import numpy as np
import matplotlib.pyplot as plt

# activation function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_gradient(x):
	return x * (1 - x)

# train the NN
def train(x, y, params):

	# characteristics of data / network
	M = x.shape[0]
	NUM_LAYERS = params["num_layers"]

	# theta values are stored in matrices where a[i] = theta[i-1] * a[i-1]
	# theta dimensions = output x input = a[i] x a[i-1]
	# starting values are random between 0 and 1
	theta = [0] * NUM_LAYERS

	for i in range(0, NUM_LAYERS - 1):
		theta[i] = np.random.rand(params["num_nodes"][i + 1], params["num_nodes"][i] + 1)

	# variables to hold intermediate values
	theta_grad = [0] * NUM_LAYERS
	a = [0] * NUM_LAYERS
	z = [0] * NUM_LAYERS
	d = [0] * NUM_LAYERS

	# initialize training variables
	a[0] = np.insert(x, 0, 1, axis=1)
	error = 100
	num_iters = 0
	MAX_ITERS = 3

	# gradient checking, to be disabled once algorithm is confirmed to be working
	GRADIENT_CHECKING = True
	epsilon = .0001
	a_plus = [0] * NUM_LAYERS 	# based on theta + epsilon
	a_minus = [0] * NUM_LAYERS 	# based on theta - epsilon

	# train the NN
	while error > params["error_limit"] and num_iters < MAX_ITERS:

		# propagate forward
		for i in range(1, NUM_LAYERS):
			
			# calculate node values
			z[i] = np.matmul(a[i - 1], theta[i - 1].T)
			a[i] = sigmoid(z[i])

			if GRADIENT_CHECKING:
				if i == 1:
					a_plus[i] = sigmoid(np.matmul(a[0], (theta[0] + epsilon).T))
					a_minus[i] = sigmoid(np.matmul(a[0], (theta[0] - epsilon).T))
				else:
					a_plus[i] = sigmoid(np.matmul(a_plus[i - 1], theta[i - 1].T))
					a_minus[i] = sigmoid(np.matmul(a_minus[i - 1], theta[i - 1].T))

			# insert bias units on all but last layer
			if i < NUM_LAYERS - 1:
				a[i] = np.insert(a[i], 0, 1, axis=1)

				if GRADIENT_CHECKING:
					a_plus[i] = np.insert(a_plus[i], 0, 1, axis=1)
					a_minus[i] = np.insert(a_minus[i], 0, 1, axis=1)

		# compute unregularized cost
		h = a[NUM_LAYERS - 1]
		cost = np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))

		if GRADIENT_CHECKING:
			grad_approx = (np.sum(-y * np.log(a_plus[NUM_LAYERS - 1]) - (1 - y) * np.log(1 - a_plus[NUM_LAYERS - 1])) - 
				np.sum(-y * np.log(a_minus[NUM_LAYERS - 1]) - (1 - y) * np.log(1 - a_minus[NUM_LAYERS - 1])) /
				2 * epsilon)

		# calculate error for output layer
		d[NUM_LAYERS - 1] = (h - y) * sigmoid_gradient(z[NUM_LAYERS - 1])

		# propagate backward
		for i in range(NUM_LAYERS - 2, -1, -1):

			# calculate error and gradients
			d[i] = np.matmul(d[i + 1], theta[i][:, 1:]) * sigmoid_gradient(z[i])
			theta_grad[i] = np.matmul(d[i + 1].T, a[i]) / M

			print(i)
			print(d[i])
			print(theta_grad[i])

			# regularize theta gradients and cost
			theta_grad[i] += np.insert(theta[i][:, 1:], 0, 0, axis=1) * params["lambda"]
			cost += (params["lambda"] / (2 * M)) * np.sum(theta[i] ** 2)

			# update weights
			theta[i] -= theta_grad[i]

		print("------")
		if GRADIENT_CHECKING:
			print(grad_approx)
		print(cost)
		print(h)
		print("------")

		num_iters += 1

	return theta

# check weights against test data
def test(x, y, theta, params):
	NUM_LAYERS = params["num_layers"]

	# variables to hold intermediate values
	a = [0] * NUM_LAYERS
	z = [0] * NUM_LAYERS

	# itialize training variables
	a[0] = np.insert(x, 0, 1, axis=1)

	# propagate forward
	for i in range(1, NUM_LAYERS):
							
		# calculate node values
		z[i] = np.matmul(a[i - 1], theta[i - 1].T)
		a[i] = sigmoid(z[i])

		# insert bias units on all but last layer
		if i < NUM_LAYERS - 1:
			a[i] = np.insert(a[i], 0, 1, axis=1)

	return a[NUM_LAYERS - 1]

# create data to use
M = 15
x = np.array([
	np.random.uniform(0, np.pi * 2, M),
	 np.random.uniform(0, np.pi * 2, M)]).T
y = np.array([np.sin(x[:, 0]),
	np.cos(x[:, 0] - x[:, 1])]).T

# initialize network parameters
params = {
	"num_layers": 3, # including input and output
	"num_nodes": {
		0: 2,	# number of characteristics in data i.e. nodes in input layer
		1: 2,	# number of nodes in hidden layer(s)
		2: 1	# number of categories i.e. nodes in output layer
	},
	"error_limit": .1,
	"lambda": 0
}

x = np.array([[1, 1, 0, 0], [1, 0, 1, 0]]).T
y = np.array([0, 1, 1, 0]).reshape(4, 1)

print(test(x, y, train(x, y, params), params))