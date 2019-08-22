from .st import *
from .mt import makeMask
import numpy as np, matplotlib.pyplot as plt, cv2

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

def top(pngImg, yolo_rst):
	h, w, c = pngImg.shape
	rgb = pngImg[...,:3]
	alpha = pngImg[...,-1].reshape(h, w, 1)

	if len(yolo_rst) == 6:
		top_x1 = yolo_rst[0]; top_y1 = yolo_rst[1]
		top_x2 = yolo_rst[2]; top_y2 = yolo_rst[3]
		top_h = yolo_rst[4]; top_w = yolo_rst[5]


		scale_h = int(top_h/10); scale_w = int(top_w/10)

		top_x1_ = top_x1-scale_w; top_y1_ = top_y1-scale_h
		top_x2_ = top_x2+scale_w; top_y2_ = top_y2+scale_h
		if top_x1_ < 0:top_x1_=0
		if top_y1_ < 0:top_y1_=0
		if top_x2_ > w:top_x2_=w-1
		if top_y2_ > h:top_y2_=h-1
		top = rgb[top_y1_:top_y1_+(top_y2_-top_y1_), top_x1_:top_x1_+(top_x2_-top_x1_)]
		top_alpha = alpha[top_y1_:top_y1_+(top_y2_-top_y1_), top_x1_:top_x1_+(top_x2_-top_x1_)]


		#topSkinMask, topSkinMask_h, topSkinMask_w = makeMask().top(top)
		#topSkinMask_ed = Erode_Dilate(topSkinMask)

		#top = cv2.bitwise_and(top, top, mask=cv2.bitwise_not(topSkinMask_ed))
		#top_alpha = cv2.bitwise_and(top_alpha, top_alpha, mask=cv2.bitwise_not(topSkinMask_ed)).reshape(topSkinMask_h, topSkinMask_w, 1)

		topClothesMask = Contour_(cv2.threshold(cv2.cvtColor(top.copy(), cv2.COLOR_RGB2GRAY),1,255,cv2.THRESH_BINARY)[1])


		#============= do ed again =============
		topClothesMask_ed = Erode_Dilate(topClothesMask)
		top = cv2.bitwise_and(top, top, mask=topClothesMask_ed)
		top_alpha = cv2.bitwise_and(top_alpha, top_alpha, mask=topClothesMask_ed).reshape(top.shape[0], top.shape[1], 1)
		#============= do ed again =============


		#============= detect junction =============
		top_h_, top_w_ = top.shape[:2]
		jct_h, jct_w = top_y2_-(2*top_y2-top_y2_), (top_x2_-top_x1_)
		jct = top[(top_h_-jct_h):top_h_, 0:jct_w]
		jct_alpha = top_alpha[(top_h_-jct_h):top_h_, 0:jct_w]
		jct_mask = Contour_(makeMask().junction(jct))

		top[(top_h_-jct_h):top_h_, 0:jct_w] = cv2.bitwise_and(jct, jct, mask=jct_mask)
		top_alpha[(top_h_-jct_h):top_h_, 0:jct_w] = cv2.bitwise_and(jct_alpha, jct_alpha, mask=jct_mask).reshape(jct_h, jct_w, 1)
		#============= detect junction =============


		#============= Combine =============
		final_top = np.concatenate([top, top_alpha], axis=2)
		#============= Combine =============

		show_rgb = rgb.copy()
		def putText(img, txt, pos, color):
			cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)
		putText(img=show_rgb, txt="Top(x1, y1)", pos=(top_x1_+10, top_y1_-10), color=(255, 0, 255))
		cv2.rectangle(show_rgb, (top_x1_, top_y1_), (top_x1_+(top_x2_-top_x1_), top_y1_+(top_y2_-top_y1_)), (255,0,255), 3)
		putText(img=show_rgb, txt="Junction(x1, y2)", pos=(top_x1_+10, (2*top_y2-top_y2_)-10), color=(255, 0, 0))
		cv2.rectangle(show_rgb, (top_x1_, 2*top_y2-top_y2_), (top_x2_, top_y2_), (255,0,0), 2)

		# ------  save points as json ------
		js = {}
		ori_img = {}
		ori_img["height"] = int(h); ori_img["width"] = int(w); ori_img["channel"] = int(c)
		js["image"] = ori_img

		top_js = {}
		top_js["x1"] = int(top_x1_); top_js["y1"] = int(top_y1_)
		top_js["x2"] = int(top_x1_+(top_x2_-top_x1_)); top_js["y2"] = int(top_y1_+(top_y2_-top_y1_))
		js["top"] = top_js
		# ------  save points as json ------

		#return [top, topSkinMask, topSkinMask_ed, topClothesMask, topClothesMask_ed, show_rgb, final_top, js]
		return [top, topClothesMask, topClothesMask_ed, show_rgb, final_top, js]

	else:
		print("{}".format(yolo_rst))

