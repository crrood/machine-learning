import numpy as np
import matplotlib.pyplot as plt

# activation function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# derivative of activation function
def sigmoid_gradient(x):
	return x * (1 - x)

# flatten a list of matrices
def unroll(theta):
	theta_unrolled = np.array([])
	for t in theta:
		if type(t) is not int:
			theta_unrolled = np.append(theta_unrolled, t.flatten())
	return theta_unrolled

# convert a flattened vector back to matrices
# based on number of nodes given in params
def reroll(theta_unrolled):
	cur = 0
	theta = [0] * (params["num_layers"] - 1)
	for j in range(0, params["num_layers"] - 1):
		theta[j] = theta_unrolled[cur : cur + params["num_nodes"][j + 1] * (params["num_nodes"][j] + 1)].reshape(
			params["num_nodes"][j + 1], params["num_nodes"][j] + 1)
		cur += params["num_nodes"][j + 1] * (params["num_nodes"][j] + 1)
	return theta

# train the NN
def train(x, y, params):

	# characteristics of data / network
	M = x.shape[0]
	NUM_LAYERS = params["num_layers"]

	# theta values are stored in matrices where a[i] = theta[i-1] * a[i-1]
	# theta dimensions = output x input = a[i] x a[i-1]
	theta = [0] * NUM_LAYERS

	# randomize starting values to break symmetry
	for i in range(0, NUM_LAYERS - 1):
		theta[i] = np.random.rand(params["num_nodes"][i + 1], params["num_nodes"][i] + 1) - .5

	theta = [np.array([[3., -2., -2.], [-1., 2., 2.]]), np.array([[-1., 2., 2.]])]

	# variables to hold intermediate values
	theta_grad = [0] * NUM_LAYERS
	a = [0] * NUM_LAYERS
	z = [0] * NUM_LAYERS
	d = [0] * NUM_LAYERS

	# initialize training variables
	a[0] = np.insert(x, 0, 1, axis=1)
	error = 100
	num_iters = 0
	MAX_ITERS = 5

	# gradient checking, to be disabled once the backpropagation algorithm is confirmed to be working
	GRADIENT_CHECKING = True
	epsilon = .0001

	# train the NN
	while error > params["error_limit"] and num_iters < MAX_ITERS:

		if GRADIENT_CHECKING:

			# convert theta to a vector for easier manipulation
			theta_unrolled = unroll(theta)
			
			# initialize lists
			N = theta_unrolled.size
			theta_plus_unrolled = [0] * N
			theta_minus_unrolled = [0] * N
			theta_plus = [0] * N
			theta_minus = [0] * N

			a_plus = [0] * N
			a_minus = [0] * N

			# create matrices
			for i in range(0, N):

				# add or subtract epsilon from unrolled theta
				theta_plus_unrolled[i] = np.copy(theta_unrolled)
				theta_plus_unrolled[i][i] += epsilon
				theta_minus_unrolled[i] = np.copy(theta_unrolled)
				theta_minus_unrolled[i][i] -= epsilon

				# prepare node matrices
				a_plus[i] = [0] * NUM_LAYERS
				a_minus[i] = [0] * NUM_LAYERS
				a_plus[i][0] = np.insert(x, 0, 1, axis=1)
				a_minus[i][0] = np.insert(x, 0, 1, axis=1)

				# convert theta_plus and theta_minus back to matrices for later use
				theta_plus[i] = reroll(theta_plus_unrolled[i])
				theta_minus[i] = reroll(theta_minus_unrolled[i])

		# propagate forward
		for i in range(1, NUM_LAYERS):
			
			# calculate node values
			z[i] = np.matmul(a[i - 1], theta[i - 1].T)
			a[i] = sigmoid(z[i])

			if GRADIENT_CHECKING:
				for j in range(0, N):
					a_plus[j][i] = sigmoid(np.matmul(a_plus[j][i - 1], theta_plus[j][i - 1].T))
					a_minus[j][i] = sigmoid(np.matmul(a_minus[j][i - 1], theta_minus[j][i - 1].T))

			# insert bias units on all but last layer
			if i < NUM_LAYERS - 1:
				a[i] = np.insert(a[i], 0, 1, axis=1)

				if GRADIENT_CHECKING:
					for j in range(0, N):
						a_plus[j][i] = np.insert(a_plus[j][i], 0, 1, axis=1)
						a_minus[j][i] = np.insert(a_minus[j][i], 0, 1, axis=1)

		# compute unregularized cost
		h = a[NUM_LAYERS - 1]
		cost = np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))

		if GRADIENT_CHECKING:
			grad_approx = [0] * N
			for i in range(0, N):
				grad_approx[i] = ((np.sum(-y * np.log(a_plus[i][NUM_LAYERS - 1]) - (1 - y) * np.log(1 - a_plus[i][NUM_LAYERS - 1])) - 
					np.sum(-y * np.log(a_minus[i][NUM_LAYERS - 1]) - (1 - y) * np.log(1 - a_minus[i][NUM_LAYERS - 1]))) /
					(2 * epsilon))

		# calculate error for output layer
		d[NUM_LAYERS - 1] = (h - y)

		# propagate backward
		for i in range(NUM_LAYERS - 2, -1, -1):

			# calculate error and gradients
			d[i] = np.matmul(d[i + 1], theta[i][:, 1:]) * sigmoid_gradient(z[i])
			theta_grad[i] = np.matmul(d[i + 1].T, a[i]) / M

			# regularize theta gradients and cost
			theta_grad[i] += np.insert(theta[i][:, 1:], 0, 0, axis=1) * params["lambda"] / M
			cost += (params["lambda"] / (2 * M)) * np.sum(theta[i] ** 2)

			# update weights
			theta[i] -= theta_grad[i]

		# use gradient checking to train weights
		# theta_unrolled = unroll(theta)
		# theta_unrolled -= grad_approx
		# theta = reroll(theta_unrolled)

		print("------")
		if GRADIENT_CHECKING:
			print(grad_approx)
			print(unroll(theta_grad))
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

	print(theta[0])
	print(a[1])

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

# test with random starting weights and backpropagation
print(test(x, y, train(x, y, params), params))

# manually weighted XOR network
# print(test(x, y, [np.array([[3, -2, -2], [-1, 2, 2]]), np.array([-1, 2, 2])], params = {
# 	"num_layers": 3, # including input and output
# 	"num_nodes": {
# 		0: 2,	# number of characteristics in data i.e. nodes in input layer
# 		1: 2,	# number of nodes in hidden layer(s)
# 		2: 1	# number of categories i.e. nodes in output layer
# 	}
# }))