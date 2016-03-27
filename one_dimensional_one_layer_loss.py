import numpy as np 
import benchmarks

LEARNING_RATE = 0.05
REGULARIZATION = 0.05
NEURON_NUM = 10

x = np.random.rand(NEURON_NUM) * 20 - 10
W = np.random.rand(NEURON_NUM)
b = np.random.rand(NEURON_NUM)

benchmark = benchmarks.cos()


# def loss(Y):
# 	return 1. * np.sum(map(lambda y: benchmark.getValue(y), Y)) / len(Y) + REGULARIZATION * np.sum(W**2)
def loss(vals):
	return 1. * np.sum(vals) / len(vals) + REGULARIZATION * np.sum(W**2)

difference = np.random.choice([-1, 1], NEURON_NUM)
prior_vals = None
b_delta = None
W_delta = None

for i in xrange(1, 10001):
	# forward prop
	Y = W * x + b

	# evaluate
	current_vals = np.array(map(lambda y: benchmark.getValue(y), Y))

	# get difference
	if prior_vals is not None:
		difference = current_vals - prior_vals
	
	# caculate loss
	L = loss(current_vals)

	# backward prop
	if b_delta is not None:
		Grad_b = -1. / NEURON_NUM * np.sign(difference) * np.sign(b_delta)
		Grad_W = -1. / NEURON_NUM * np.sign(difference) * np.sign(W_delta)
	else:
		Grad_b = -1. / NEURON_NUM * difference
		Grad_W = -1. / NEURON_NUM * difference

	# calculate b_delta and W_delta
	b_delta = LEARNING_RATE * Grad_b
	W_delta = LEARNING_RATE * Grad_W

	# update the params
	b += b_delta
	W += W_delta


	# update prior_vals
	prior_vals = current_vals

	# verbose
	print "%d generation: Loss: %f" % (i, L)
print W*x + b
print np.array(map(lambda y: benchmark.getValue(y), W*x + b))