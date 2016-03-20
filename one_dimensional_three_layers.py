import numpy as np 
import benchmarks
import pprint

LEARNING_RATE = 1E-3
REGULARIZATION = 0.05

neuron_nums = [5, 10, 100]
delta = 1E-10

x = np.random.rand(neuron_nums[0]) * 20 - 10
W1 = np.random.rand(neuron_nums[0])
b1 = np.random.rand(neuron_nums[0])
W2 = np.random.rand(neuron_nums[1], neuron_nums[0])
b2 = np.random.rand(neuron_nums[1])
W3 = np.random.rand(neuron_nums[2], neuron_nums[1])
b3 = np.random.rand(neuron_nums[2])

benchmark = benchmarks.sin()

def loss(Y):
	return 1. * np.sum(map(lambda y: benchmark.getValue(y), Y)) / len(Y) + REGULARIZATION * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

for i in xrange(1, 20001):
	# forward prop
	Y1 = W1 * x + b1
	Y2 = W2.dot(Y1) + b2
	Y3 = W3.dot(Y2) + b3

	# calculate loss
	L = loss(Y3)

	# backward prop
	Grad_b3 = 1. / neuron_nums[2] * np.array(map(lambda y: benchmark.getGrad(y), Y3))

	# # grad test for b3
	# print Grad_b3[0]
	# b3[0] += delta
	# LPlusDelta = loss(W3.dot(Y2) + b3)
	# print (LPlusDelta - L) / delta

	Grad_W3 = np.repeat(Y2.reshape(1, neuron_nums[1]), neuron_nums[2], axis=0) * Grad_b3.reshape(neuron_nums[2], 1)
	
	# # grad test for W3
	# print Grad_W3[0, 0]
	# W3[0, 0] += delta
	# LPlusDelta = loss(W3.dot(Y2) + b3)
	# print (LPlusDelta - L) / delta

	Grad_b2 = np.sum(W3 * Grad_b3.reshape(neuron_nums[2], 1), axis=0)

	# # grad test for b2
	# print Grad_b2[0]
	# b2[0] += delta
	# LPlusDelta = loss(W3.dot(W2.dot(Y1) + b2) + b3)
	# print (LPlusDelta - L) / delta

	Grad_W2 = np.repeat(Y1.reshape(1, neuron_nums[0]), neuron_nums[1], axis=0) * Grad_b2.reshape(neuron_nums[1], 1)

	# # grad test for W2
	# print Grad_W2[0, 0]
	# W2[0, 0] += delta
	# LPlusDelta = loss(W3.dot(W2.dot(Y1) + b2) + b3)
	# print (LPlusDelta - L) / delta

	Grad_b1 = np.sum(W2 * Grad_b2.reshape(neuron_nums[1], 1), axis=0)

	# # grad test for b1
	# print Grad_b1[0]
	# b1[0] += delta
	# LPlusDelta = loss(W3.dot(W2.dot(W1 * x + b1) + b2) + b3)
	# print (LPlusDelta - L) / delta

	Grad_W1 = Grad_b1 * x

	# # grad test for W1
	# print Grad_W1[0]
	# W1[0] += delta
	# LPlusDelta = loss(W3.dot(W2.dot(W1 * x + b1) + b2) + b3)
	# print (LPlusDelta - L) / delta

	# update the params
	b3 -= LEARNING_RATE * Grad_b3
	W3 -= LEARNING_RATE * Grad_W3
	b2 -= LEARNING_RATE * Grad_b2
	W2 -= LEARNING_RATE * Grad_W2
	b1 -= LEARNING_RATE * Grad_b1
	W1 -= LEARNING_RATE * Grad_W1

	# verbose
	print "%d generation: Loss: %f" % (i, L)

pprint.pprint(W3.dot(W2.dot(W1 * x + b1) + b2) + b3)
pprint.pprint(map(lambda y: benchmark.getValue(y), W3.dot(W2.dot(W1 * x + b1) + b2) + b3))