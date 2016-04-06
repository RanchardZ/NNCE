from network2 import UnitNetwork
from copy import copy
from benchmarks import Sphere, Rosenbrock, Rastrigin
from layer import EvaluateLayer2
from params import initUniformTwoDimArray
from loss import averageLoss

import numpy as np
sphere = Sphere()
rosenbrock = Rosenbrock()

popNum = 20
dimNum = 50

problemConfig = {
	"popNum": popNum,
	"dimNum": dimNum,
	"bound": [-10, 10],
	"benchmark": Rosenbrock,
	"structure": [popNum, popNum, popNum],
	"iterationNum": 10000,
	"learningRate": 1E-5,
}

class Web(object):

	def __init__(self, config):
		self.web = self.initWeb(config)
		self.evl = self.initEvl(config)
		self.inputs = self.initInputs(config)
		self.outputs = np.zeros_like(self.inputs)
		self.diffsBack = np.zeros(config["popNum"])
		self.iterationNum = config["iterationNum"]

	@staticmethod
	def initWeb(config):
		structure = config['structure']
		popNum = config['popNum']
		dimNum = config['dimNum']
		learningRate = config['learningRate']
		web = []
		unitConfig = {
			"structure": structure,
			"learningRate": learningRate
		}
		for i in range(dimNum):
			web.append(UnitNetwork(unitConfig))
		return web

	@staticmethod
	def initEvl(config):
		return EvaluateLayer2(config["benchmark"])

	@staticmethod
	def initInputs(config):
		popNum, dimNum = config["popNum"], config["dimNum"]
		lower, upper = config["bound"]
		inputs = initUniformTwoDimArray(dimNum, popNum, lower, upper)
		return inputs

	def run(self):
		for i in xrange(1, self.iterationNum + 1):
			self.forward()
			fitValues = self.evaluate()
			self.backward()
			self.update()

			computedLoss = averageLoss(fitValues)
			print "%d iteration: Loss - %f" % (i, computedLoss)
		print self.evl.inputsForth
		print self.evl.curFitValues

	def forward(self):
		for i, (unitNetwork, inputsForth) in enumerate(zip(self.web, self.inputs)):
			self.outputs[i] = unitNetwork.forward(inputsForth)

	def evaluate(self):
		# for row in self.outputs:
		# 	np.random.shuffle(row)
		fitValues = self.evl.forward(self.outputs)
		self.diffsBack = self.evl.backward()
		return fitValues


	def backward(self):
		# diffsBack = np.array([diff1, diff2, diff3 ...]), len=popNum
		for unitNetwork in self.web:
			unitNetwork.backward(self.diffsBack)

	def update(self):
		for unitNetwork in self.web:
			unitNetwork.update()


