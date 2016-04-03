from layer import FullyConnectedFirstLayer, FullyConnectedHiddenLayer, FullyConnectedLastLayer, EvaluateLayer
from loss import averageLoss
from params import initUniformOneDimArray
from config import learningRate, networkStructure, benchmark, iterationNum

from copy import copy

inputs = initUniformOneDimArray(networkStructure[0], -10, 10)

network = []
network.append(FullyConnectedFirstLayer(networkStructure[0]))
for (preNueronNum, curNeuronNum) in zip(networkStructure[:-1], networkStructure[1:]):
	network.append(FullyConnectedHiddenLayer(preNueronNum, curNeuronNum))
network.append(EvaluateLayer(benchmark))

for i in xrange(1, iterationNum + 1):
	# forward propagation
	inputsForth = copy(inputs)
	for layer in network[:-1]:
		inputsForth = layer.forward(inputsForth)

	# evaluate
	fitValues = network[-1].forward(inputsForth)
	computedLoss = averageLoss(fitValues)

	# backward propagation
	diffsBack = network[-1].backward()
	for layer in reversed(network[:-1]):
		diffsBack = layer.backward(diffsBack)

	# update params()
	for layer in network[:-1]:
		layer.update()

	# verbose
	print "%d iteration: Loss - %f" % (i, computedLoss)

print network[-1].inputsForth
print network[-1].curFitValues
