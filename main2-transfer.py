# ======================================================#
#														#
#				Transfer Color and Cartoonize			#
#														#
# ======================================================#

import warnings, os
warnings.simplefilter("ignore")

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--IMAGE", required=True, help="Required Image")
ap.add_argument("-sk", "--SKIN", type=int, default=0, help="Skin Color")
ap.add_argument("-cs", "--CLOTHES", type=int, default=0, help="Clothes Color")
ap.add_argument("-ct", "--CARTOON", type=int, default=0, help="Cartoon Style")
args = vars(ap.parse_args())

import numpy as np, cv2, json
from scipy import misc
from utils import reverse, normalize

def split(img):
	return [img[...,:3], img[...,-1].reshape(img.shape[0], img.shape[1], 1)]

def read_json(jfile, keys):
	with open(jfile) as jr:
		js = json.load(jr)[keys]
		x1 = js["x1"]; x2 = js["x2"]
		y1 = js["y1"]; y2 = js["y2"]
		h = y2-y1; w = x2-x1
		return [x1, y1, x2, y2, h, w]

def cartoon(img):
	from utils import cartoon_m2

	rgb, alpha = split(img)

	import tensorflow as tf
	with tf.Session() as sess:
		print(sess.run(tf.constant("Cartoonize...")).decode())
	cartoon = cartoon_m2(rgb, n_clusters=8)
	misc.imsave(os.path.join(filename, filename+"-cartoon.png"), np.concatenate([cartoon, alpha], axis=-1))

def skin_transfer():
	from keras.models import load_model
	#headR = load_model("models/skin_color/Head_Red.h5")
	#headP = load_model("models/skin_color/Head_Purple.h5")
	#headG = load_model("models/skin_color/Head_Green.h5")

	#armLR = load_model("models/skin_color/Arms_LS_Red.h5")
	#armLP = load_model("models/skin_color/Arms_LS_Purple.h5")
	#armLG = load_model("models/skin_color/Arms_LS_Green.h5")

	#armRR = load_model("models/skin_color/Arms_RS_Red.h5")
	#armRP = load_model("models/skin_color/Arms_RS_Purple.h5")
	#armRG = load_model("models/skin_color/Arms_RS_Green.h5")

	colordict1 = {
		1:load_model("models/skin_color/Head_Red.h5"),
		2:load_model("models/skin_color/Head_Purple.h5"),
		3:load_model("models/skin_color/Head_Green.h5")
	}
	colordict2 = {
		1:load_model("models/skin_color/Arms_LS_Red.h5"),
		2:load_model("models/skin_color/Arms_LS_Purple.h5"),
		3:load_model("models/skin_color/Arms_LS_Green.h5")
	}
	colordict3 = {
		1:load_model("models/skin_color/Arms_RS_Red.h5"),
		2:load_model("models/skin_color/Arms_RS_Purple.h5"),
		3:load_model("models/skin_color/Arms_RS_Green.h5")
	}

	colortype_q = int(input("Paint Type (Same 0, Differet 1): "))

	if colortype_q == 0:
		colorskin = int(input("Skin Color (Red 1, Purple 2, Green 3): "))
		#colorclothes = int(input("Clothes (Blue 1, Red 2):"))
	elif colortype_q == 1:
		print("Random color")
		colorskin = 4
	else:
		print("Wrong Argument: Type only for 0 or 1.")
		exit()

	if colorskin > 0 and colorskin <= 3:
		c1_ = colordict1[colorskin]
		c2_ = colordict2[colorskin]
		c3_ = colordict3[colorskin]
	elif colorskin == 4:
		c1_ = colordict1[np.random.randint(1,4)]
		c2_ = colordict2[np.random.randint(1,4)]
		c3_ = colordict3[np.random.randint(1,4)]
	else:
		print("Wrong Argument: SkinColor only for 1, 2, 3 and 4.")
		exit()

	jpgImg = misc.imread(os.path.join(filename+".jpg"))
	h,w,c = jpgImg.shape

	head = misc.imread(os.path.join(filename, filename+"-head.png"))
	armL = misc.imread(os.path.join(filename, filename+"-armL.png"))
	armR = misc.imread(os.path.join(filename, filename+"-armR.png"))

	color_head = cv2.resize(reverse(c1_.predict(normalize(head))), (w,h))
	color_armL = cv2.resize(reverse(c2_.predict(normalize(armL))), (w,h))
	color_armR = cv2.resize(reverse(c3_.predict(normalize(armR))), (w,h))

	#if h>500 or w>500:
	#	nh = nw = 500
	#	jpgImg = cv2.resize(jpgImg, (nh,nw))
	#	color_head = cv2.resize(color_head, (nh,nw))
	#	color_armL = cv2.resize(color_armL, (nh,nw))
	#	color_armR = cv2.resize(color_armR, (nh,nw))
	#else:
	#	nh = h
	#	nw = w

	for h_ in range(h):
		for w_ in range(w):
			if np.mean(color_head[h_][w_][...,3]) > 127:
				jpgImg[h_][w_] = color_head[h_][w_][...,:3]
			elif np.mean(color_armL[h_][w_][...,3]) > 127:
				jpgImg[h_][w_] = color_armL[h_][w_][...,:3]
			elif np.mean(color_armR[h_][w_][...,3]) > 127:
				jpgImg[h_][w_] = color_armR[h_][w_][...,:3]
			else:
				continue

	#misc.imsave(os.path.join(filename, filename+"-test.jpg"), cv2.resize(jpgImg, (w,h)))
	misc.imsave(os.path.join(filename, filename+"-color-skin.jpg"), jpgImg)

def clothes_transfer():
	from keras.models import load_model

	top_model = load_model("models/Top_cb.h5")
	bot_model = load_model("models/Bottom_cb.h5")

	jpgImg = misc.imread(os.path.join(filename+".jpg"))
	h,w,c = jpgImg.shape

	top = misc.imread(os.path.join(filename, filename+"-top.png"))
	bot = misc.imread(os.path.join(filename, filename+"-bottom.png"))

	jf1 = os.path.join(filename, filename+"-top.json")
	jf2 = os.path.join(filename, filename+"-bottom.json")

	tx1, ty1, tx2, ty2, th, tw = read_json(jf1, "top")
	bx1, by1, bx2, by2, bh, bw = read_json(jf2, "bottom")

	color_top = cv2.resize(reverse(top_model.predict(normalize(np.concatenate([top, top], axis=-1)))), (tw,th))
	color_bot = cv2.resize(reverse(bot_model.predict(normalize(np.concatenate([bot, bot], axis=-1)))), (bw,bh))

	colortype_q = int(input("Paint Type (Same 0, Differet 1): "))

	if colortype_q == 0:
		colorclothes = int(input("Clothes Color (Blue 1, Red 2): "))
	elif colortype_q == 1:
		print("Random color")
		colorclothes = 3
	else:
		print("Wrong Argument: Type only for 0 or 1.")
		exit()

	colordict1 = {1:color_top[...,:4], 2:color_top[...,4:]}
	colordict2 = {1:color_bot[...,:4], 2:color_bot[...,4:]}

	if colorclothes > 0 and colorclothes <= 2:
		c1_ = colordict1[colorclothes]
		c2_ = colordict2[colorclothes]
	elif colorclothes == 3:
		c1_ = colordict1[np.random.randint(1,3)]
		c2_ = colordict2[np.random.randint(1,3)]
	else:
		print("Wrong Argument: SkinColor only for 1 or 2.")
		exit()


	top_h, top_w = c1_.shape[:2]

	for h_ in range(top_h):
		for w_ in range(top_w):
			if np.mean(c1_[h_][w_][...,3]) > 127:
				jpgImg[ty1+h_, tx1+w_] = c1_[h_][w_][...,:3]
			else:
				continue

	bot_h, bot_w = c2_.shape[:2]

	for h_ in range(bot_h):
		for w_ in range(bot_w):
			if np.mean(c2_[h_][w_][...,3]) > 127:
				jpgImg[by1+h_, bx1+w_] = c2_[h_][w_][...,:3]
			else:
				continue

	misc.imsave(os.path.join(filename, filename+"-color-clothes.jpg"), jpgImg)

if __name__ == '__main__':
	filename = args["IMAGE"].split(".")[0]
	if (args["SKIN"] < 0 or args["SKIN"] > 1) and (args["CARTOON"] < 0 or args["CARTOON"] > 0) and (args["CLOTHES"] < 0 or args["CLOTHES"] > 0):
		print("Wrong Argument: -sk, -cs and -ct only for 0 or 1.")
		exit()

	try:
		img = misc.imread(os.path.join(filename, filename+"-nobg.png"))
	except Exception as e:
		print(e)
		exit()

	if args["CARTOON"]:
		cartoon(img)

	if args["SKIN"]:
		skin_transfer()

	if args["CLOTHES"]:
		clothes_transfer()
