import cv2, numpy as np

def normalize(img):
	return cv2.resize(img, (256,256)).reshape(1,256,256,img.shape[-1])/127.5-1

def reverse(pred):
	return np.uint8(pred[0]*127.5+127.5)