import numpy as np



LEARNING_RATE = 0.05
REGULARIZATION = 0.05
NEURON_NUM = 3

x = np.random.rand()
W = np.random.rand(NEURON_NUM)
b = np.random.rand(NEURON_NUM)
# x = -52.8692174238
# W = np.array([0.7944, 0.5633, 0.9764])
# b = np.array([0.0498, 0.0256, 0.5681])


def benchmark(y):
	# return y**2 - 1.
	return np.sin(y)

def loss(Y):
	return 1. * np.sum(map(lambda y: benchmark(y), Y)) / len(Y) + REGULARIZATION * np.sum(W**2)

for i in xrange(1, 1001):
	# forward prop
	Y = W * x + b

	# calculate loss
	L = loss(Y)

	# backward prop
	# Grad_b = -1. / NEURON_NUM * 2 * Y 
	# Grad_b = -1. / NEURON_NUM * np.cos(Y)
	Grad_b = 1. / NEURON_NUM * np.sign(np.sin(Y))
	# Grad_b = -1. / NEURON_NUM * np.sign(np.cos(Y))
	Grad_W = Grad_b * x

	# update the params
	b += LEARNING_RATE * Grad_b
	W += LEARNING_RATE * Grad_W

	# verbose
	print "%d generation: Loss: %f" % (i, L)
print W*x + b