import numpy
import tkinter as tk
from PIL import ImageGrab, Image
from snn import SNN
from mnist import imageNormalizer

def getLabel(outputs):
	index = numpy.argmax(outputs)
	return index if outputs[index] > 0.4 else "unknown"

class Paint():
	def __init__(self):
		self.nn = SNN("mnist-784-100-10.txt")

		self.root = tk.Tk()
		self.bgcolor = 'black'
		self.fcolor = 'white'
		self.width = 20

		self.query = tk.Button(self.root, text = 'query', command = self.query)
		self.query.grid(row = 0, column = 0)

		self.result = tk.StringVar()
		self.resultLabel = tk.Label(self.root, textvariable = self.result)
		self.resultLabel.grid(row = 0, column = 1)		

		self.erase = tk.Button(self.root, text = 'erase', command = self.erase)
		self.erase.grid(row = 0, column = 2)

		self.canvas = tk.Canvas(self.root, bg = self.bgcolor, width = 300, height = 300)
		self.canvas.grid(row = 1, columnspan = 3)
		self.canvas.bind('<B1-Motion>', self.paint)
		self.canvas.bind('<ButtonRelease-1>', self.reset)
		
		self.x = None
		self.y = None

		self.root.mainloop()

	def paint(self, event):
		if self.x and self.y:
			self.canvas.create_line(self.x, self.y, event.x, event.y, 
				width = self.width, fill = self.fcolor, capstyle = tk.ROUND)
		self.x = event.x
		self.y = event.y

	def reset(self, event):
		self.x = None
		self.y = None

	def query(self):
		x = self.root.winfo_rootx() + self.canvas.winfo_x()
		y = self.root.winfo_rooty() + self.canvas.winfo_y()
		x1 = x + self.canvas.winfo_width()
		y1 = y + self.canvas.winfo_height()
		resized = ImageGrab.grab(bbox = (x + 4, y + 4, x1 - 4, y1 - 4)).resize((28, 28), resample = Image.BILINEAR);
		outputs = self.nn.query(imageNormalizer(list(resized.getdata(0))))[-1]
		self.result.set(getLabel(outputs))

	def erase(self):
		self.canvas.delete("all")

Paint()