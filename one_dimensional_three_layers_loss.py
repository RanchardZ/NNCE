import numpy as np 
import benchmarks
import pprint, sys

LEARNING_RATE = 1E-4
REGULARIZATION = 0.05

neuron_nums = [5, 20, 10]
delta = 1E-10

x = np.random.rand(neuron_nums[0]) * 20 - 10
W1 = np.random.rand(neuron_nums[0])
b1 = np.random.rand(neuron_nums[0])
W2 = np.random.rand(neuron_nums[1], neuron_nums[0])
b2 = np.random.rand(neuron_nums[1])
W3 = np.random.rand(neuron_nums[2], neuron_nums[1])
b3 = np.random.rand(neuron_nums[2])

difference = np.random.choice([-1, 1], neuron_nums[2])
prior_vals = None
b3_delta = None
W3_delta = None
b2_delta = None
W2_delta = None
b1_delta = None
W1_delta = None

benchmark = benchmarks.sin()

def loss(vals):
	return 1. * np.sum(vals) / len(vals) + REGULARIZATION * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

for i in xrange(1, 10001):
	# forward prop
	Y1 = W1 * x + b1
	Y2 = W2.dot(Y1) + b2
	Y3 = W3.dot(Y2) + b3

	# evaluate
	current_vals = np.array(map(lambda y: benchmark.getValue(y), Y3))

	# get difference
	if prior_vals is not None:
		difference = current_vals - prior_vals

	# calculate loss
	L = loss(current_vals)

	# backward prop
	if b3_delta is not None:
		Grad_b3 = -1. / neuron_nums[2] * np.sign(difference) * np.sign(b3_delta)
	else:
		Grad_b3 = -1. / neuron_nums[2] * difference
	Grad_W3 = -np.repeat(Y2.reshape(1, neuron_nums[1]), neuron_nums[2], axis=0) * Grad_b3.reshape(neuron_nums[2], 1)
	Grad_b2 = np.sum(W3 * Grad_b3.reshape(neuron_nums[2], 1), axis=0)
	Grad_W2 = np.repeat(Y1.reshape(1, neuron_nums[0]), neuron_nums[1], axis=0) * Grad_b2.reshape(neuron_nums[1], 1)
	Grad_b1 = np.sum(W2 * Grad_b2.reshape(neuron_nums[1], 1), axis=0)
	Grad_W1 = Grad_b1 * x

	# calculate deltas
	b3_delta = LEARNING_RATE * Grad_b3
	W3_delta = LEARNING_RATE * Grad_W3
	b2_delta = LEARNING_RATE * Grad_b2
	W2_delta = LEARNING_RATE * Grad_W2
	b1_delta = LEARNING_RATE * Grad_b1
	W1_delta = LEARNING_RATE * Grad_W1

	# update the params
	b3 += b3_delta
	W3 += W3_delta
	b2 += b2_delta
	W2 += W2_delta
	b1 += b1_delta
	W1 += W1_delta

	# update prior_vals
	prior_vals = current_vals

	# verbose
	print "%d generation: Loss: %f" % (i, L)
print W3.dot(W2.dot(W1 * x + b1) + b2) + b3
print np.array(map(lambda y: benchmark.getValue(y), W3.dot(W2.dot(W1 * x + b1) + b2) + b3))