import numpy as np

class benchmark(object):
	def __init__(self):
		pass

class parabola(benchmark):
	@staticmethod
	def getGrad(x):
		return 2*x
	@staticmethod
	def getValue(x):
		return x**2

class sin(benchmark):
	@staticmethod
	def getGrad(x):
		return np.cos(x)
	@staticmethod
	def getValue(x):
		return np.sin(x)

class cos(benchmark):
	@staticmethod
	def getGrad(x):
		return -np.sin(x)
	@staticmethod
	def getValue(x):
		return np.cos(x)