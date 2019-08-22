# ======================================================#
#														#
#						Rebuild							#
#														#
# ======================================================#

import warnings, os
warnings.simplefilter("ignore")

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--IMGDIR", required=True, help="Required Image Dir")
args = vars(ap.parse_args())

from scipy import misc
import numpy as np, cv2

jpgImg = misc.imread(os.path.join(args["IMGDIR"]+".jpg"))
h,w,c = jpgImg.shape

color_head = misc.imread(os.path.join(args["IMGDIR"], args["IMGDIR"]+"-head-color.png"))
color_armL = misc.imread(os.path.join(args["IMGDIR"], args["IMGDIR"]+"-armL-color.png"))
color_armR = misc.imread(os.path.join(args["IMGDIR"], args["IMGDIR"]+"-armR-color.png"))

jpgImg = cv2.resize(jpgImg, (500,500))
color_head = cv2.resize(color_head, (500,500))
color_armL = cv2.resize(color_armL, (500,500))
color_armR = cv2.resize(color_armR, (500,500))

for h_ in range(500):
	for w_ in range(500):
		if np.mean(color_head[h_][w_][...,3]) > 0:
			jpgImg[h_][w_] = color_head[h_][w_][...,:3]
		elif np.mean(color_armL[h_][w_][...,3]) > 0:
			jpgImg[h_][w_] = color_armL[h_][w_][...,:3]
		elif np.mean(color_armR[h_][w_][...,3]) > 0:
			jpgImg[h_][w_] = color_armR[h_][w_][...,:3]
		else:
			continue

misc.imsave(os.path.join(args["IMGDIR"], args["IMGDIR"]+"-test.jpg"), cv2.resize(jpgImg, (h,w)))