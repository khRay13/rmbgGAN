import time
def getTime():
	return time.strftime("%H:%M:%S", time.localtime())
from scipy import misc
from tools.mask_tool import mask_tool as mt, makeMask as mk
import numpy as np, json, requests
import matplotlib.pyplot as plt, cv2


def top(imgname, save_top = 1, save_mask = 1):
	print("[System {}]: cropTop Start".format(getTime()))
	ts = time.time()
	img_ = misc.imread(imgname+".png")
	h, w, c = img_.shape
	rgb = img_[...,:3].copy()
	alpha = img_[...,-1].reshape(h, w, 1).copy()

	#===============  YOLO API  ===============
	print("[Top {}]: Connect YOLO Server".format(getTime()))
	url = 'http://163.18.42.200:508/yolo/top'
	content_type = 'image/jpeg'
	headers = {'content-type': content_type}
	_state_, imgCode = cv2.imencode('.jpg', rgb)
	response = requests.post(url, data=imgCode.tostring(), headers=headers, timeout = 3)
	result = json.loads(response.text)
	#print(result)
	#===============  YOLO API  ===============

	if len(result) == 6:
		top_x1 = result[0]; top_y1 = result[1]
		top_x2 = result[2]; top_y2 = result[3]
		top_h = result[4]; top_w = result[5]


		scale_h = int(top_h/10); scale_w = int(top_w/10)

		print("[Top {}]: Crop top region".format(getTime()))
		top_x1_ = top_x1-scale_w; top_y1_ = top_y1-scale_h
		top_x2_ = top_x2+scale_w; top_y2_ = top_y2+scale_h
		if top_x1_ < 0:top_x1_=0
		if top_y1_ < 0:top_y1_=0
		if top_x2_ > w:top_x2_=w-1
		if top_y2_ > h:top_y2_=h-1
		top = rgb[top_y1_:top_y1_+(top_y2_-top_y1_), top_x1_:top_x1_+(top_x2_-top_x1_)]
		misc.imsave("cut/"+imgname+"-top_crop.jpg", top)
		top_alpha = alpha[top_y1_:top_y1_+(top_y2_-top_y1_), top_x1_:top_x1_+(top_x2_-top_x1_)]


		print("[Top {}]: Make top skin mask".format(getTime()))
		topSkinMask, topSkinMask_h, topSkinMask_w = mk().top(top)
		topSkinMask_ed = mt().erode_dilate(topSkinMask)
		topClothesMask = topSkinMask_ed.copy()

		top = cv2.bitwise_and(top, top, mask=cv2.bitwise_not(topSkinMask_ed))
		top_alpha = cv2.bitwise_and(top_alpha, top_alpha, mask=cv2.bitwise_not(topSkinMask_ed)).reshape(topSkinMask_h, topSkinMask_w, 1)

		topClothesMask = mt().Contour_deNoise(cv2.threshold(cv2.cvtColor(top.copy(), cv2.COLOR_RGB2GRAY),1,255,cv2.THRESH_BINARY)[1])


		#============= do ed again =============
		topClothesMask_ed = mt().erode_dilate(topClothesMask)
		top = cv2.bitwise_and(top, top, mask=topClothesMask_ed)
		top_alpha = cv2.bitwise_and(top_alpha, top_alpha, mask=topClothesMask_ed).reshape(topSkinMask_h, topSkinMask_w, 1)
		#============= do ed again =============


		#============= detect junction =============
		print("[Top {}]: Detect junction".format(getTime()))
		top_h_, top_w_ = top.shape[:2]
		jct_h, jct_w = top_y2_-(2*top_y2-top_y2_), (top_x2_-top_x1_)
		jct = top[(top_h_-jct_h):top_h_, 0:jct_w]
		jct_alpha = top_alpha[(top_h_-jct_h):top_h_, 0:jct_w]
		print("[Top {}]: Make junction mask".format(getTime()))
		jct_mask = mt().Contour_deNoise(mk().junction(jct))

		top[(top_h_-jct_h):top_h_, 0:jct_w] = cv2.bitwise_and(jct, jct, mask=jct_mask)
		top_alpha[(top_h_-jct_h):top_h_, 0:jct_w] = cv2.bitwise_and(jct_alpha, jct_alpha, mask=jct_mask).reshape(jct_h, jct_w, 1)
		#============= detect junction =============


		#============= Combine =============
		print("[Top {}]: Save top as {}-top.png".format(getTime(), imgname))
		result = np.concatenate([top, top_alpha], axis=2)
		misc.imsave("cut/"+imgname+"-top.png", result)
		#============= Combine =============

		if save_mask:
			print("[Top {}]: Save mask as {}-top-Mask.jpg".format(getTime(), imgname))
			smR,smC = 2,2
			cnt = 0
			titles = ["skin", "skin e&d", "clothes", "clothes e&d"]
			masks = [topSkinMask, topSkinMask_ed, topClothesMask, topClothesMask_ed]
			for sr_ in range(smR*smC):
				cnt+=1
				plt.subplot(smR, smC, cnt)
				plt.title(titles[sr_])
				plt.imshow(masks[sr_], cmap="gray")
				plt.axis("off")
			#plt.show()
			plt.savefig("cut/"+imgname+"-topMask.jpg")

			#cv2.imwrite(imgname+"topMask-skin-row.jpg", topSkinMask)
			#cv2.imwrite(imgname+"topMask-skin-ed.jpg", topSkinMask_ed)
			#cv2.imwrite(imgname+"topMask-noise-row.jpg", topClothesMask)
			#cv2.imwrite(imgname+"topMask-noise-ed.jpg", topClothesMask_ed)

		if save_top:
			print("[Top {}]: Save region as {}-top-region.jpg".format(getTime(), imgname))
			show_rgb = img_[...,:3].copy()
			def putText(img, txt, pos, color):
				cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)
			putText(img=show_rgb, txt="Top(x1, y1)", pos=(top_x1_+10, top_y1_-10), color=(255, 0, 255))
			cv2.rectangle(show_rgb, (top_x1_, top_y1_), (top_x1_+(top_x2_-top_x1_), top_y1_+(top_y2_-top_y1_)), (255,0,255), 3)
			putText(img=show_rgb, txt="Junction(x1, y2)", pos=(top_x1_+10, (2*top_y2-top_y2_)-10), color=(255, 0, 0))
			cv2.rectangle(show_rgb, (top_x1_, 2*top_y2-top_y2_), (top_x2_, top_y2_), (255,0,0), 2)
			misc.imsave("cut/"+imgname+"-top-region.jpg", show_rgb)
			#plt.imshow(show_rgb); plt.show()


		# ------  save points as json ------
		print("[Head {}]: Save points as {}-top-point.json".format(getTime(), imgname))
		js = {}
		ori_img = {}
		ori_img["height"] = int(h); ori_img["width"] = int(w); ori_img["channel"] = int(c)
		js["image"] = ori_img

		top_js = {}
		top_js["x1"] = int(top_x1); top_js["y1"] = int(top_y1)
		top_js["x2"] = int(top_x2); top_js["y2"] = int(top_y2)
		top_js["height"] = int(top_h); top_js["width"] = int(top_w);
		top_js["scale_h"] = int(scale_h); top_js["scale_w"] = int(scale_w)
		js["face"] = top_js

		#print(js)
		with open("cut/"+imgname+"-top-point.json", "w") as jw:
			json.dump(js, jw)
		# ------  save points as json ------

	else:
		print("[Top {}]: {}".format(getTime(), result))

	print("[System {}]: cropTop Finish, Cost: {:.5f}sec\n".format(getTime(), time.time()-ts))


#if __name__ == '__main__':
#	top(imgname = "img_00000207-nobg.png".split(".")[0])
