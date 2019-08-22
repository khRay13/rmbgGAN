#from scipy import misc
#from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import cv2#, numpy as np

def cartoon_m1(img):
	#img = misc.imread(imgname)
	#rgb = img[...,:3]
	#alpha = img[...,-1].reshape(img.shape[0], img.shape[1], 1)
	#print(alpha.shape)

	# 1) Edges
	#gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gray = cv2.medianBlur(gray, 3)
	edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)

	# 2) Color
	color = cv2.bilateralFilter(img, 9, 300, 300)

	# 3) Cartoon
	cartoon = cv2.bitwise_and(color, color, mask=edges)


	#cv2.imshow("Image", cv2.cvtColor(np.concatenate([rgb, alpha], axis=-1), cv2.COLOR_RGB2BGR))
	#cv2.imshow("Cartoon", cv2.cvtColor(np.concatenate([cartoon, alpha], axis=-1), cv2.COLOR_RGB2BGR))
	#cv2.imshow("edges", edges)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return cartoon

def cartoon_m2(img, n_clusters=8):
	#img = misc.imread(imgname)
	(h, w) = img.shape[:2]
	#rgb = img[...,:3]
	#alpha = img[...,-1].reshape(img.shape[0], img.shape[1], 1)

	# 1) Edges
	#gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gray = cv2.medianBlur(gray, 3)
	edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 7)

	# 2) Color
	imgLab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape((h * w, 3))
	clt = MiniBatchKMeans(n_clusters = n_clusters)
	labels = clt.fit_predict(imgLab)
	quant = (clt.cluster_centers_.astype("uint8")[labels]).reshape((h, w, 3))
	color = quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)


	# 3) Cartoon
	cartoon = cv2.bitwise_and(color, color, mask=edges)
	#cartoon_ = np.concatenate([cartoon, alpha], axis=-1)

	#misc.imsave(imgname+"-cartoon.png", cartoon_)
	return cartoon

#if __name__ == '__main__':
#	method2("img_00000003-nobg.png", n_clusters=8)