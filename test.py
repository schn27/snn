import numpy
from mnist import imageNormalizer
from mnist import forEach
from snn import SNN

def getLabel(outputs):
	return numpy.argmax(outputs)

class Processor:
	def __init__(self):
		self.nn = SNN("mnist-784-100-10.txt")
		self.error = 0
		self.total = 0

	def process(self, label, image):
		self.total += 1
		queryLabel = getLabel(self.nn.query(imageNormalizer(image))[-1])
		
		if queryLabel != label:
			self.error += 1

	def finish(self):
		print("error rate: ", (self.error * 100.0) / self.total)

forEach("data/t10k-labels.idx1-ubyte", "data/t10k-images.idx3-ubyte", 10000, Processor())
