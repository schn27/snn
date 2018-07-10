import numpy
import io

def imageNormalizer(image):
	return [x / 255.0 * 0.99 + 0.01 for x in image]

def targetsFromLabel(label):
	targets = numpy.zeros(10) + 0.01
	targets[label] = 0.99
	return targets	

def forEach(labelsFileName, imagesFileName, n, processor):
	imgSize = 28 * 28

	with open(labelsFileName, "rb") as f:
		f.read(8)
		labels = f.read(n)

	with open(imagesFileName, "rb") as f:
		f.read(16)
		images = f.read(imgSize * n)

	for i in range(n):
		processor.process(labels[i], images[imgSize * i : imgSize * (i + 1)])
		if (i % 1000) == 0:
			print(i)

	processor.finish()
