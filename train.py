from mnist import forEach
from mnist import imageNormalizer
from mnist import targetsFromLabel
from snn import SNN

class Processor:
	def __init__(self):
		self.nn = SNN([784, 100, 10])

	def process(self, label, image):
		self.nn.train(imageNormalizer(image), targetsFromLabel(label), 0.1)

	def finish(self):
		self.nn.dumpToFile("mnist-784-100-10.txt")		

forEach("data/train-labels.idx1-ubyte", "data/train-images.idx3-ubyte", 60000, Processor())


