# A simple neural network implementation

import numpy
import scipy.special
import io

class SNN:
	def __init__(self, config):
		self.activationFunc = lambda x: scipy.special.expit(x)
		
		if type(config) == str:
			self.loadFromFile(config)
		else:
			self.weights = [];
			previousLayer = 0;

			for layer in config:
				if previousLayer > 0:
					self.weights.append(numpy.random.normal(0.0, pow(previousLayer, -0.5), (layer, previousLayer)))

				previousLayer = layer

	def train(self, inputList, targetList, learningGrade):
		targets = numpy.array(targetList, ndmin = 2).T
		outputs = self.query(inputList)

		error = targets - outputs[-1]

		for i in range(len(self.weights)):
			w = self.weights[-i - 1]
			nextError = numpy.dot(w.T, error)
			out = outputs[-i - 1]
			prev = outputs[-i - 2]
			w += learningGrade * numpy.dot(error * out * (1.0 - out), prev.T)
			error = nextError

	def query(self, inputList):
		outputs = [numpy.array(inputList, ndmin = 2).T]
		
		for w in self.weights:
			outputs.append(self.activationFunc(numpy.dot(w, outputs[-1])))

		return outputs

	def dumpToFile(self, fileName):
		with open(fileName, "w") as file:
			print(len(self.weights), file = file)
			
			for w in self.weights:
				r, c = w.shape
				print(r, file = file)
				print(c, file = file)
				
				for row in w.tolist():
					for e in row:
						print(e, file = file)

	def loadFromFile(self, fileName):
		with open(fileName, "r") as file:
			self.weights = []

			for i in range(int(file.readline())):
				r = int(file.readline())
				c = int(file.readline())
				w = []

				for j in range(r * c):
					w.append(float(file.readline()))

				self.weights.append(numpy.asfarray(w).reshape(r, c))
