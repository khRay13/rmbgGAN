from .st import *
from .mt import *
import numpy as np, cv2, json, requests

def Erode_Dilate(mask):
	def process(mask_, kernel, iters):
		top_erode1 = cv2.erode(mask_, kernel, iterations=iters)
		top_dilate = cv2.dilate(top_erode1, kernel, iterations=iters*2)
		top_erode2 = cv2.erode(top_dilate, kernel, iterations=iters)
		return top_erode2

	p1 = process(mask, np.ones([3,3], dtype=np.uint8), iters=2)
	p2 = process(p1, np.ones([7,7], dtype=np.uint8), iters=1)

	return p2

def Contour_(mask):
	msk = mask.copy()
	(_, cnts, _) = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	area = []
	for c_ in cnts:area.append(cv2.contourArea(c_))
	for a_ in range(len(area)):
		if a_ != np.argmax(area):msk = cv2.drawContours(msk, [cnts[a_]], -1, 0, -1)
	return msk

def bottom(pngImg, yolo_rst):
	h,w,c = pngImg.shape
	rgb = pngImg[...,:3]
	alpha = pngImg[...,-1]

	if len(yolo_rst) == 6:
		bot_x1 = yolo_rst[0]; bot_y1 = yolo_rst[1]
		bot_x2 = yolo_rst[2]; bot_y2 = yolo_rst[3]
		bot_h = yolo_rst[4]; bot_w = yolo_rst[5]

		#===============  Save region figure  ===============
		botRegion = rgb.copy()
		cv2.putText(botRegion, "Bottom", (bot_x1+10, bot_y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,0,255), 1, cv2.LINE_AA)
		cv2.rectangle(botRegion, (bot_x1, bot_y1), (bot_x2, bot_y2), (255,0,255), 3)
		#===============  Save region figure  ===============


		#===============  Crop bottom region  ===============
		bias = int(bot_h/5)
		bot = rgb[bot_y1-bias:bot_y2, bot_x1:bot_x2]
		bot_alpha = alpha[bot_y1-bias:bot_y2, bot_x1:bot_x2]
		#===============  Crop bottom region  ===============


		#===============  Remove skin  ===============
		#mask1 = skintool().detect_skin(bot)
		#mask_h, mask_w = mask1.shape
		#if mask1[int(mask_h/2)][int(mask_w/2)] == 0: mask1 = masktool().reverse_area(mask1)
		#ed1 = Erode_Dilate(mask1)
		#bot_nsk = cv2.bitwise_and(bot, bot, mask=ed1)
		#===============  Remove skin  ===============


		#===============  Denoise #1  ===============
		#_, mask2 = cv2.threshold(cv2.cvtColor(bot_nsk.copy(), cv2.COLOR_RGB2GRAY),1,255,cv2.THRESH_BINARY)
		#ed2 = Erode_Dilate(mask2)
		#bot = cv2.bitwise_and(bot, bot, mask=ed2)
		#bot_alpha = cv2.bitwise_and(bot_alpha, bot_alpha, mask=ed2)
		#===============  Denoise #1  ===============


		#===============  Process junction area  ===============
		jct_h, jct_w = 2*bias, bot_w
		jct = bot[0:jct_h, 0:jct_w]
		jct_alpha = bot_alpha[0:jct_h, 0:jct_w]

		jctmsk = makeMask().junction(jct)
		if jctmsk[0][int(jct_w/2)] == 255: jctmsk = masktool().reverse_area(jctmsk)

		bot[0:jct_h, 0:jct_w] = cv2.bitwise_and(jct, jct, mask=jctmsk)
		bot_alpha[0:jct_h, 0:jct_w] = cv2.bitwise_and(jct_alpha, jct_alpha, mask=jctmsk)
		#===============  Process junction area  ===============


		#===============  Denoise #2  ===============
		_, mask3 = cv2.threshold(cv2.cvtColor(bot.copy(), cv2.COLOR_RGB2GRAY),1,255,cv2.THRESH_BINARY)
		ed3 = Erode_Dilate(mask3)
		CdeN = Contour_(ed3)
		bot = cv2.bitwise_and(bot, bot, mask=CdeN)
		bot_alpha = cv2.bitwise_and(bot_alpha, bot_alpha, mask=CdeN)
		#=============== Denoise #2  ===============


		#===============  Combine #1  ===============
		final_bot = np.concatenate([bot, bot_alpha.reshape(bot_h+bias, bot_w, 1)], axis=2)
		#===============  Combine #1  ===============


		#===============  Save point as json  ===============
		js = {}
		ori_img = {}
		ori_img["height"] = int(h); ori_img["width"] = int(w); ori_img["channel"] = int(c)
		js["image"] = ori_img

		bot_js = {}
		bot_js["x1"] = int(bot_x1); bot_js["y1"] = int(bot_y1-bias)
		bot_js["x2"] = int(bot_x2); bot_js["y2"] = int(bot_y2)
		bot_js["height"] = int(bot_h+bias); bot_js["width"] = int(bot_w);
		js["bottom"] = bot_js
		#===============  Save point as json  ===============

		#return [bot, mask2, mask3, ed2, ed3, jct, jctmsk, botRegion, final_bot, js]
		return [bot, mask3, ed3, jct, jctmsk, botRegion, final_bot, js]

	else:
		print("{}".format(yolo_rst))
