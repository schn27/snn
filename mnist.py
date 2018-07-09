import numpy
import io

def imageNormalizer(image):
	return [x / 255.0 * 0.99 + 0.01 for x in image]

def targetsFromLabel(label):
	targets = numpy.zeros(10) + 0.01
	targets[label] = 0.99
	return targets	

def forEach(labelsFileName, imagesFileName, n, processor):
	with open(labelsFileName, "rb") as labels:
		with open(imagesFileName, "rb") as images:
			labels.read(8)			
			images.read(16)

			for i in range(n):
				processor.process(ord(labels.read(1)), images.read(28 * 28))
				if (i % 1000) == 0:
					print(i)

			processor.finish()
