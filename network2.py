from layer import FullyConnectedFirstLayer, FullyConnectedHiddenLayer
from copy import copy

# template for config file
# config = {
# 	"structure": [5, 5, 5],
# 	"bound": [-10, 10]
# }

# class UnitNetwork(object):

# 	def __init__(self, config):
# 		self.network = self.initNetwork(config)
	
# 	@staticmethod
# 	def initNetwork(config):
# 		learningRate = config["learningRate"]
# 		structure = config["structure"]
# 		network = []
# 		network.append(FullyConnectedFirstLayer(structure[0], learningRate))			
# 		for (preNueronNum, curNeuronNum) in zip(structure[:-1], structure[1:]):
# 			network.append(FullyConnectedHiddenLayer(preNueronNum, curNeuronNum, learningRate))
# 		return network

# 	def forward(self, inputsForth):
# 		# forward propagation
# 		for layer in self.network:
# 			inputsForth = layer.forward(inputsForth)
# 		return inputsForth

# 	def backward(self, diffsBack):
# 		# backward propagation
# 		for layer in reversed(self.network):
# 			diffsBack = layer.backward(diffsBack)

# 	def update(self):
# 		for layer in self.network:
# 			layer.update()

class NetworkUnit(object):

	def __init__(self, structure, learningRate):
		self.structure = structure
		self.learningRate = learningRate

		self.network = self.initNetwork()

	def initNetwork(self):
		network = []
		network.append(FullyConnectedFirstLayer(self.structure[0], self.learningRate))
		for (preNueronNum, curNeuronNum) in zip(self.structure[:-1], self.structure[1:]):
			network.append(FullyConnectedHiddenLayer(preNueronNum, curNeuronNum, self.learningRate))
		return network

	def forward(self, inputsForth):
		# forward propagation
		for layer in self.network:
			inputsForth = layer.forward(inputsForth)
		return inputsForth

	def backward(self, diffsBack):
		# backward propagation
		for layer in reversed(self.network):
			diffsBack = layer.backward(diffsBack)

	def update(self):
		for layer in self.network:
			layer.update()