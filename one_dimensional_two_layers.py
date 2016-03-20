import numpy as np 
import benchmarks

LEARNING_RATE = 1E-3
REGULARIZATION = 0.05
FIRST_LAYER_NEURON_NUM = 5
SECOND_LAYER_NEURON_NUM = 5
delta = 1E-10

x = np.random.rand(FIRST_LAYER_NEURON_NUM) * 20 - 10
W1 = np.random.rand(FIRST_LAYER_NEURON_NUM)
b1 = np.random.rand(FIRST_LAYER_NEURON_NUM)
W2 = np.random.rand(SECOND_LAYER_NEURON_NUM, FIRST_LAYER_NEURON_NUM)
b2 = np.random.rand(SECOND_LAYER_NEURON_NUM)

benchmark = benchmarks.parabola()

def loss(Y):
	return 1. * np.sum(map(lambda y: benchmark.getValue(y), Y)) / len(Y) + REGULARIZATION * np.sum(W1**2 + W2**2)

for i in xrange(1, 1001):
	# forward prop
	Y1 = W1 * x + b1
	Y2 = W2.dot(Y1) + b2

	# calculate loss
	L = loss(Y2)

	# backward prop
	Grad_b2 = 1. / SECOND_LAYER_NEURON_NUM * np.array(map(lambda y: benchmark.getGrad(y), Y2))
	

	# # grad test for b2
	# print Grad_b2[2]
	# b2[2] += delta
	# LPlusDelta = loss(W2.dot(Y1) + b2)
	# print (LPlusDelta - L) / delta

	Grad_W2 = np.repeat(Y1.reshape(1, FIRST_LAYER_NEURON_NUM), SECOND_LAYER_NEURON_NUM, axis=0) * Grad_b2.reshape(SECOND_LAYER_NEURON_NUM, 1)

	# # grad test for W2
	# print Grad_W2[0, 1]
	# W2[0, 1] += delta
	# LPlusDelta = loss(W2.dot(Y1) + b2)
	# print (LPlusDelta - L) / delta

	Grad_b1 = np.sum(W2 * Grad_b2.reshape(SECOND_LAYER_NEURON_NUM, 1), axis=0)

	# # grad test for b1
	# print Grad_b1[1]
	# b1[1] += delta
	# LPlusDelta = loss(W2.dot(W1 * x + b1) + b2)
	# print (LPlusDelta - L) / delta


	Grad_W1 = Grad_b1 * x

	# # grad test for W1
	# print Grad_W1[0]
	# W1[0] += delta
	# LPlusDelta = loss(W2.dot(W1 * x + b1) + b2)
	# print (LPlusDelta - L) / delta

	# update the params
	b2 -= LEARNING_RATE * Grad_b2
	W2 -= LEARNING_RATE * Grad_W2
	b1 -= LEARNING_RATE * Grad_b1
	W2 -= LEARNING_RATE * Grad_W1

	# verbose
	print "%d generation: Loss: %f" % (i, L)
# print W1
# print W2
print W2.dot(W1*x + b1) + b2
# print map(lambda y: benchmark.getValue(y), W2.dot(W1*x + b1) + b2)
