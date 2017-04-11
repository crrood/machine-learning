import numpy as np
import matplotlib.pyplot as plt

# activation function
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

# train the NN
def train(x, y, params):
	# theta values are stored in matrices where a[i] = theta[i-1] * a[i-1]
	# with dimensions = output x input = a[i] x a[i-1]
	theta = []

	# initialize random weights and variables to hold node values
	a = []
	z = []
	for i in range(0, params["num_layers"] - 1):
		theta.append(np.random.rand(params["num_nodes"][i + 1], params["num_nodes"][i]))
		a.append(0)
		z.append(0)

	print(theta)

	# propagate forward
	# and train weights
	error = 100
	a[0] = np.insert(x, 0, 0, axis=1)
	while error > params["error_limit"]:
		for i in range(1, params["num_layers"] + 1):
			z[i] = np.matmul(a[i-1], theta[i-1])
			a[i] = sigmoid(z[i])
		error = np.sum(a[i] - y)
		error = 0

	print(a)

	# gradient checking

# create data to use
x = np.array([np.random.uniform(0, np.pi * 2, 50), np.random.uniform(0, np.pi * 2, 50)]).T
y = np.array([np.sin(x[:,0]), np.cos(x[:,0] - x[:,1])]).T

# initialize network parameters
params = {
	"num_layers": 3, # including input and output
	"num_nodes": {
		0: 2, # number of characteristics in data i.e. nodes in input layer
		1: 10,
		2: 2
	},
	"lambda": 0,
	"error_limit": 10
}

train(x, y, params)