from network2 import NetworkUnit
from benchmarks import Sphere, Rosenbrock
from layer import RandomEvaluateLayer
from params import initUniformTwoDimArray
from loss import averageLoss

from copy import copy
import numpy as np 

# sphere = Sphere()
# rosenbrock = Rosenbrock()

# popNum = 3
# dimNum = 5

class randomCombinedWeb(object):

	def __init__(self, popNum, dimNum, bound, benchmark, unitStructure, iterNum, learningRate):
		self.popNum = popNum
		self.dimNum = dimNum
		self.bound = bound
		self.unitStructure = unitStructure
		self.iterNum = iterNum
		self.learningRate = learningRate

		self.web = self.initWeb()
		self.evl = RandomEvaluateLayer(benchmark)
		self.inputs = self.initInputs()
		self.outputs = np.zeros_like(self.inputs)
		self.diffsBack = np.zeros(popNum)

	def initWeb(self):
		web = []
		for i in range(self.dimNum):
			web.append(NetworkUnit(self.unitStructure, self.learningRate))
		return web

	def initInputs(self):
		lower, upper = self.bound
		inputs = initUniformTwoDimArray(self.dimNum, self.popNum, lower, upper)
		return inputs

	def run(self):
		mappedFitValues = []
		for i in xrange(1, self.iterNum + 1):
			self.forward()
			mappedFitValues = self.evaluate()
			self.backward()
			self.update()

			computedLoss = averageLoss(mappedFitValues)
			print "%d iteration: Loss - %f" % (i, computedLoss)
		print self.evl.inputsForth
		print mappedFitValues	

	def forward(self):
		for i, (networkUnit, inputForth) in enumerate(zip(self.web, self.inputs)):
			self.outputs[i] = networkUnit.forward(inputForth)

	def evaluate(self):
		fitValues = self.evl.forward(self.outputs)
		self.diffsBack = self.evl.backward()
		return fitValues

	def backward(self):
		for (networkUnit, diffsBack) in zip(self.web, self.diffsBack):
			networkUnit.backward(diffsBack)

	def update(self):
		for unitNetwork in self.web:
			unitNetwork.update()
		# self.inputs = self.initInputs()


