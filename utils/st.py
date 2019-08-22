import warnings as ws
ws.simplefilter("ignore")
import cv2, numpy as np

class skintool():
	def __init__(self):
		super(skintool, self).__init__()

	def detect_skin(self, img):
		ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
		(y, cr, cb) = cv2.split(ycrcb)

		cr1 = cv2.GaussianBlur(cr, (3, 3), 0)
		_, skin = cv2.threshold(cr1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		kernel = np.ones([5,5], dtype=np.uint8)
		erode = cv2.erode(cv2.dilate(skin, kernel, iterations=2), kernel, iterations=2)

		return erode

	def detect_skin2(self, img):
		h, w, c = img.shape
		R, G, B = cv2.split(img)
		Y = R*0.299+G*0.587+B*0.114
		Cb = (R*-0.169+G*-0.331+B*0.5)+128
		Cr = (R*0.5+G*-0.419+B*-0.081)+128


		skinMask = cv2.GaussianBlur(Cr, (3, 3), 0)

		_, skin = cv2.threshold(np.uint8(skinMask),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		return skin

	def detect_skin3(self, img):
		h, w, c = img.shape
		R, G, B = cv2.split(img)

		Y = cv2.GaussianBlur(R*0.299+G*0.587+B*0.114, (3, 3), 0)
		Cb = cv2.GaussianBlur((R*-0.169+G*-0.331+B*0.5)+128, (3, 3), 0)
		Cr = cv2.GaussianBlur((R*0.5+G*-0.419+B*-0.081)+128, (3, 3), 0)

		Ymin = 16; Ymax = 235
		Crmin = 133; Crmax = 173
		Cbmin = 77; Cbmax = 127

		mask = np.zeros([h, w], dtype=np.uint8)
		for h_ in range(h):
			for w_ in range(w):
				R_ = R[h_][w_]; G_ = G[h_][w_]; B_ = B[h_][w_]
				Y_ = Y[h_][w_]; Cr_ = Cr[h_][w_]; Cb_ = Cb[h_][w_]

				if (
					(Y_>Ymin and Y_<Ymax and Cb_>Cbmin and Cb_<Cbmax and Cr_>Crmin and Cr_<Crmax) and
					(R_>G_ and R_>B_ and (R_-G_)>15)
				):
					mask[h_][w_] = 255

		return mask

	def detect_skin4(self, img):
		h, w, c = img.shape
		mask = np.zeros([h, w], dtype=np.uint8)

		from tools import cvtHSV
		print("split to R G B")
		R,G,B = cv2.split(img)
		print("covert to H S V")
		H,S,V = cv2.split(cvtHSV.cvtRGB2HSV(img))

		for h_ in range(h):
			for w_ in range(w):
				R_ = R[h_][w_]; G_ = G[h_][w_]; B_ = B[h_][w_]
				H_ = H[h_][w_]; S_ = S[h_][w_]; V_ = V[h_][w_]

				if (
					(H_>=0 and H_<=50) and
					(S_>=0.23 and S_<=0.68) and
					(R_>G_) and
					(R_>B_) and
					((R_-G_)>15)
				):
					mask[h_][w_] = 255

		return mask
