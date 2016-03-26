import numpy as np 
import benchmarks

LEARNING_RATE = 1E-3
REGULARIZATION = 0.05
FIRST_LAYER_NEURON_NUM = 5
SECOND_LAYER_NEURON_NUM = 10
delta = 1E-10

benchmark = benchmarks.cos()

x = np.random.rand(FIRST_LAYER_NEURON_NUM) * 20 - 10
W1 = np.random.rand(FIRST_LAYER_NEURON_NUM)
b1 = np.random.rand(FIRST_LAYER_NEURON_NUM)
W2 = np.random.rand(SECOND_LAYER_NEURON_NUM, FIRST_LAYER_NEURON_NUM)
b2 = np.random.rand(SECOND_LAYER_NEURON_NUM)

difference = np.random.choice([-1, 1], SECOND_LAYER_NEURON_NUM)
prior_vals = None
b2_delta = None
W2_delta = None
b1_delta = None
W1_delta = None

def loss(vals):
	return 1. * np.sum(vals) / len(vals) + REGULARIZATION * (np.sum(W1**2) + np.sum(W2**2))


for i in xrange(1, 10001):
	# forward prop
	Y1 = W1 * x + b1
	Y2 = W2.dot(Y1) + b2

	# evaluate
	current_vals = np.array(map(lambda y: benchmark.getValue(y), Y2))

	# get difference
	if prior_vals is not None:
		difference = current_vals - prior_vals

	# calculate loss
	L = loss(current_vals)

	# backward prop
	if b2_delta is not None:
		Grad_b2 = 1. / SECOND_LAYER_NEURON_NUM * np.sign(difference) * np.sign(b2_delta)
		Grad_W2 = np.repeat(Y1.reshape(1, FIRST_LAYER_NEURON_NUM), SECOND_LAYER_NEURON_NUM, axis=0) * Grad_b2.reshape(SECOND_LAYER_NEURON_NUM, 1) * np.sign(Grad_W2)
	else:
		Grad_b2 = 1. / SECOND_LAYER_NEURON_NUM * difference
		Grad_W2 = np.repeat(Y1.reshape(1, FIRST_LAYER_NEURON_NUM), SECOND_LAYER_NEURON_NUM, axis=0) * Grad_b2.reshape(SECOND_LAYER_NEURON_NUM, 1)

	if b1_delta is not None:
		Grad_b1 = 1. * np.sum(W2 * Grad_b2.reshape(SECOND_LAYER_NEURON_NUM, 1), axis=0) * np.sign(b1_delta)
		Grad_W1 = Grad_b1 * x * np.sign(W1_delta)
	else:
		Grad_b1 = 1. * np.sum(W2 * Grad_b2.reshape(SECOND_LAYER_NEURON_NUM, 1), axis=0)
		Grad_W1 = Grad_b1 * x
	

	# calculate deltas
	b2_delta = -LEARNING_RATE * Grad_b2
	W2_delta = -LEARNING_RATE * Grad_W2
	b1_delta = -LEARNING_RATE * Grad_b1
	W1_delta = -LEARNING_RATE * Grad_W1

	# update the params
	b2 += b2_delta
	W2 += W2_delta
	b1 += b1_delta
	W1 += W1_delta

	# update prior_vals
	prior_vals = current_vals

	# verbose
	print "%d generation: Loss: %f" % (i, L)
print W2.dot(W1*x + b1) + b2
print np.array(map(lambda y: benchmark.getValue(y), W2.dot(W1*x + b1) + b2))


