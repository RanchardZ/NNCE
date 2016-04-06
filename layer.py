import numpy as np 
from params import initUniformOneDimArray, initUniformTwoDimArray
# from config import learningRate

class FullyConnectedHiddenLayer(object):

	def __init__(self, preNeuronNum, curNeuronNum, learningRate, lower = -0.5, upper = 0.5):
		self.learningRate = learningRate
		self.preNeuronNum = preNeuronNum
		self.curNeuronNum = curNeuronNum
		
		self.b = initUniformOneDimArray(curNeuronNum, lower, upper)
		self.W = initUniformTwoDimArray(curNeuronNum, preNeuronNum, lower, upper)

	def forward(self, inputsForth):
		assert(len(inputsForth) == self.preNeuronNum)

		self.inputsForth = inputsForth
		return self.W.dot(inputsForth) + self.b

	def backward(self, diffsBack):
		assert(len(diffsBack) == self.curNeuronNum)

		reshapedDiffsBack = diffsBack.reshape(self.curNeuronNum, 1)

		try:
			self.gradsOfb = np.sign(diffsBack) * np.sign(self.deltaOfb)
			self.gradsOfW = np.sign(reshapedDiffsBack) * np.sign(self.deltaOfW)
		except AttributeError:
			self.gradsOfb = np.sign(diffsBack)
			self.gradsOfW = np.sign(reshapedDiffsBack) * np.sign(self.W)

		return np.sum(1. / self.curNeuronNum * np.sign(reshapedDiffsBack) * np.sign(self.W) , axis = 0)



	def update(self):
		self.deltaOfb = -self.learningRate * self.gradsOfb
		self.deltaOfW = -self.learningRate * self.gradsOfW

		self.b += self.deltaOfb
		self.W += self.deltaOfW


class FullyConnectedFirstLayer(object):

	def __init__(self, neuronNum, learningRate, lower = -0.5, upper = 0.5):
		self.learningRate = learningRate
		self.neuronNum = neuronNum

		self.b = initUniformOneDimArray(neuronNum, lower, upper)
		self.W = initUniformOneDimArray(neuronNum, lower, upper)

	def forward(self, inputsForth):
		assert(len(inputsForth) == self.neuronNum)

		self.inputsForth = inputsForth
		return self.W * inputsForth + self.b

	def backward(self, diffsBack):
		assert(len(diffsBack) == self.neuronNum)

		try:
			self.gradsOfb = np.sign(diffsBack) * np.sign(self.deltaOfb)
			self.gradsOfW = np.sign(diffsBack) * np.sign(self.deltaOfW)
		except AttributeError:
			self.gradsOfb = np.sign(diffsBack)
			self.gradsOfW = np.sign(diffsBack) * np.sign(self.W)
		return

	def update(self):
		self.deltaOfb = -self.learningRate * self.gradsOfb
		self.deltaOfW = -self.learningRate * self.gradsOfW

		self.b += self.deltaOfb
		self.W += self.deltaOfW

class FullyConnectedLastLayer(FullyConnectedHiddenLayer):

	def __init__(self, preNeuronNum, curNeuronNum, learningRate):
		super(FullyConnectedLastLayer, self).__init__(preNeuronNum, curNeuronNum, learningRate)


class EvaluateLayer(object):

	def __init__(self, benchmark):
		self.benchmark = benchmark

	def forward(self, inputsForth):
		self.inputsForth = inputsForth
		self.curFitValues = np.array(map(lambda x: self.benchmark.evaluate(x), inputsForth))
		try:
			self.diffsBack = self.curFitValues - self.priFitValues
		except:
			self.diffsBack = np.random.choice([-1, 1], len(inputsForth))
		self.priFitValues = self.curFitValues
		return self.curFitValues

	def backward(self):
		return self.diffsBack

class EvaluateLayer2(object):

	def __init__(self, benchmark):
		self.benchmark = benchmark

	def forward(self, inputsForth):
		self.inputsForth = inputsForth
		self.curFitValues = np.array(map(lambda x: self.benchmark.evaluate(x), inputsForth.T))
		try:
			self.diffsBack = self.curFitValues - self.priFitValues
		except:
			self.diffsBack = np.random.choice([-1, 1], len(self.curFitValues))
		self.priFitValues = self.curFitValues
		return self.curFitValues

	def backward(self):
		return self.diffsBack
