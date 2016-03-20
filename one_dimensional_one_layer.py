import numpy as np
import benchmarks



LEARNING_RATE = 0.05
REGULARIZATION = 0.05
NEURON_NUM = 5

x = np.random.rand(NEURON_NUM) * 20 - 10
W = np.random.rand(NEURON_NUM)
b = np.random.rand(NEURON_NUM)

benchmark = benchmarks.parabola()

def loss(Y):
	return 1. * np.sum(map(lambda y: benchmark.getValue(y), Y)) / len(Y) + REGULARIZATION * np.sum(W**2)

for i in xrange(1, 1001):
	# forward prop
	Y = W * x + b

	# calculate loss
	L = loss(Y)

	# backward prop
	Grad_b = -1. / NEURON_NUM * np.sign(map(lambda y: benchmark.getGrad(y), Y))
	Grad_W = Grad_b * x

	# update the params
	b += LEARNING_RATE * Grad_b
	W += LEARNING_RATE * Grad_W

	# verbose
	print "%d generation: Loss: %f" % (i, L)
print W*x + b