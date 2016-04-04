import numpy as np

class oneDimensionalBenchmark(object):
	def __init__(self):
		pass

class parabola(oneDimensionalBenchmark):
	@staticmethod
	def getGrad(x):
		return 2*x
	@staticmethod
	def evaluate(x):
		return x**2

class sin(oneDimensionalBenchmark):
	@staticmethod
	def getGrad(x):
		return np.cos(x)
	@staticmethod
	def evaluate(x):
		return np.sin(x)

class cos(oneDimensionalBenchmark):
	@staticmethod
	def getGrad(x):
		return -np.sin(x)
	@staticmethod
	def evaluate(x):
		return np.cos(x)

class benchmark(object):
	def __init__(self):
		pass

	@staticmethod
	def evaluate(x):
		raise NotImplementedError

class Sphere(benchmark):
	def __init__(self):
		super(Sphere, self).__init__()

	@staticmethod
	def evaluate(x):
		return np.sum(np.square(x))

class Rastrigin(benchmark):
	def __init__(self):
		super(Rastrigin, self).__init__()

	@staticmethod
	def evaluate(x):
		return 10. * len(x) + np.sum(np.square(x) - 10. * np.cos(x * 2 * np.pi))

class Rosenbrock(benchmark):
	def __init__(self):
		super(Rosenbrock, self).__init__()

	@staticmethod
	def evaluate(x):
		return np.sum(100. * np.square(x[1:] - np.square(x[:-1])) + np.square(1 - x[:-1]))