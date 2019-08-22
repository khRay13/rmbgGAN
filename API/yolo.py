import warnings as ws
ws.simplefilter("ignore")
from darkflow.net.build import TFNet

class yolo():
	def __init__(self, ops):
		self.ops = ops

	def createNet(self):
		return TFNet(self.ops)