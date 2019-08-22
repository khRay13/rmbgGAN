from scipy import misc
from .st import skintool
import matplotlib.pyplot as plt
import numpy as np, cv2

class masktool():
	def __init__(self):
		super(masktool, self).__init__()

	def binary(self, mask, ratio):
		mask_ = mask.copy()
		h,w = mask_.shape[:2]
		for h_ in range(h):
			for w_ in range(w):
				if mask_[h_][w_] > ratio:
					mask_[h_][w_] = 0
				else:
					mask_[h_][w_] = 255
		return mask_

	def reverse_area(self, roi):
		return np.uint8(roi*(-1)+255)

	def chk_point(self, pot):
		if pot == 255:
			return True #should be reverse area
		else:
			return False

class noisetool():
	def __init__(self):
		super(noisetool, self).__init__()

	def deNoise1_er(self, mask, kernel_size = 3, iters = 1):
		def erSET(mask, kernel, iters = iters):
			msk = cv2.erode(mask, kernel, iterations=iters)
			msk = cv2.dilate(msk, kernel, iterations=iters)
			return msk
		def erSET_r(mask, kernel, iters = iters):
			msk = cv2.dilate(mask, kernel, iterations=iters)
			msk = cv2.erode(msk, kernel, iterations=iters)
			return msk

		h,w = mask.shape[:2]

		mask_ = erSET(mask.copy(), np.ones([kernel_size,kernel_size], dtype=np.uint8))
		mask_ = self.deNoise2_contour(mask_, 0)
		mask_ = self.deNoise2_contour(mask_, 255)
		mask_ = erSET_r(mask_, np.ones([3,3], dtype=np.uint8))
		return mask_

	def deNoise2_contour(self, mask, fill_color):
		msk = mask.copy()
		#(_, cnts, _) = cv2.findContours(msk, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		(_, cnts, _) = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		area = []
		for c_ in cnts:area.append(cv2.contourArea(c_))
		for a_ in range(len(area)):
			if a_ != np.argmax(area): msk = cv2.drawContours(msk, [cnts[a_]], -1, fill_color, -1)
		return msk

class makeMask():
	def __init__(self):
		super(makeMask, self).__init__()
		self.mt = masktool()
		self.st = skintool()

	def top(self, img):
		topMask = self.st.detect_skin2(img)
		topMask_h, topMask_w = topMask.shape
		if self.mt.chk_point(topMask[int(topMask_h/2)][int(topMask_w/2)]):
			topMask = self.mt.reverse_area(topMask)

		return topMask, topMask_h, topMask_w

	def junction(self, img, kernel=np.ones([3,3], dtype=np.uint8), iters=2):
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		g = cv2.GaussianBlur(gray, (5, 5), 0)
		_, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		if mask[0][int(mask.shape[1]/2)] == 0:
			mask = self.mt.reverse_area(mask)
		erode1 = cv2.dilate(mask, kernel, iterations=iters)
		dilate = cv2.erode(erode1, kernel, iterations=iters*2)
		erode2 = cv2.dilate(dilate, kernel, iterations=iters)

		return erode2

#if __name__ == '__main__':
#	img = misc.imread("sample6-mask.jpg")
#	mask = tools().deNoise1_er(tools().binary(img))
#	plt.imshow(mask, cmap="gray");plt.show()
