import warnings as ws
ws.simplefilter("ignore")
from darkflow.net.build import TFNet

class yolonet():
	def __init__(self, ops):
		self.ops = ops

	def createNet(self):
		return TFNet(self.ops)

	def parsing(self, js, keys):
		detected = False
		for r_ in js:
			if r_["label"] == keys:
				detected = True
				x1 = r_["topleft"]["x"]; y1 = r_["topleft"]["y"]
				x2 = r_["bottomright"]["x"]; y2 = r_["bottomright"]["y"]
				h = y2 - y1; w = x2 - x1

		if not detected:
			return "Haven't detected."
		else:
			return [x1, y1, x2, y2, h, w]