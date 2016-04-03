import numpy as np 

def initUniformOneDimArray(dim, lower, upper):
	return np.random.rand(dim) * (upper - lower) + lower

def initUniformTwoDimArray(axisZeroDim, axisOneDim, lower, upper):
	return np.random.rand(axisZeroDim, axisOneDim) * (upper - lower) + lower