from mnist import forEach
from mnist import imageNormalizer
from mnist import targetsFromLabel
from snn import SNN

class Processor:
	def __init__(self, config):
		self.config = config
		self.nn = SNN(config)

	def process(self, label, image):
		self.nn.train(imageNormalizer(image), targetsFromLabel(label), 0.2)

	def finish(self):
		name = "mnist"
		for v in self.config:
			name += "-" + str(v)
		name += ".txt"
		self.nn.dumpToFile(name)

processor = Processor([784, 100, 10])

for i in range(3):
	print("epoch:", i + 1)
	forEach("data/train-labels.idx1-ubyte", "data/train-images.idx3-ubyte", 60000, processor)
