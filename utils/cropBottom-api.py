import time
def getTime():
	return time.strftime("%H:%M:%S", time.localtime())
from scipy import misc
from tools.mask_tool import mask_tool as mt
from tools.mask_tool import makeMask as mk
from matplotlib import pyplot as plt
import numpy as np, cv2, json, requests


def Bottom(imgname, save_bottom = 1, save_mask = 1, save_jct = 1):
	print("[System {}]: cropBottom Start".format(getTime()))
	ts = time.time()
	img = misc.imread(imgname+".png")
	h,w,c = img.shape
	rgb = img[...,:3]
	alpha = img[...,-1].copy()


	#===============  YOLO API  ===============
	print("[Bottom {}]: Connect YOLO Server".format(getTime()))
	url = 'http://163.18.42.200:508/yolo/bottom'
	content_type = 'image/jpeg'
	headers = {'content-type': content_type}
	_state_, imgCode = cv2.imencode('.jpg', rgb)
	response = requests.post(url, data=imgCode.tostring(), headers=headers, timeout = 3)
	result = json.loads(response.text)
	#print(result)
	#===============  YOLO API  ===============


	if len(result) == 6:
		bot_x1 = result[0]; bot_y1 = result[1]
		bot_x2 = result[2]; bot_y2 = result[3]
		bot_h = result[4]; bot_w = result[5]


		#===============  Save region figure  ===============
		botRegion = rgb.copy()
		cv2.putText(botRegion, "Bottom", (bot_x1+10, bot_y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,0,255), 1, cv2.LINE_AA)
		cv2.rectangle(botRegion, (bot_x1, bot_y1), (bot_x2, bot_y2), (255,0,255), 3)
		misc.imsave("cut/"+imgname+"-bottom-region.jpg", botRegion)
		#===============  Save region figure  ===============


		#===============  Crop bottom region  ===============
		print("[Bottom {}]: Crop bottom region".format(getTime()))
		bias = int(bot_h/5)
		bot = rgb[bot_y1-bias:bot_y2, bot_x1:bot_x2]
		misc.imsave("cut/"+imgname+"-bottom_crop.jpg", bot)
		bot_alpha = alpha[bot_y1-bias:bot_y2, bot_x1:bot_x2]
		#===============  Crop bottom region  ===============


		#===============  Remove skin  ===============
		mask1 = mt().detect_skin(bot)
		mask_h, mask_w = mask1.shape
		if mask1[int(mask_h/2)][int(mask_w/2)] == 0: mask1 = mt().reverse_area(mask1)
		ed1 = mt().erode_dilate(mask1)
		bot_nsk = cv2.bitwise_and(bot, bot, mask=ed1)
		#===============  Remove skin  ===============


		#===============  Denoise #1  ===============
		_, mask2 = cv2.threshold(cv2.cvtColor(bot_nsk.copy(), cv2.COLOR_RGB2GRAY),1,255,cv2.THRESH_BINARY)
		ed2 = mt().erode_dilate(mask2)
		bot = cv2.bitwise_and(bot, bot, mask=ed2)
		bot_alpha = cv2.bitwise_and(bot_alpha, bot_alpha, mask=ed2)
		#===============  Denoise #1  ===============


		#===============  Process junction area  ===============
		jct_h, jct_w = 2*bias, bot_w
		jct = bot[0:jct_h, 0:jct_w]
		jct_alpha = bot_alpha[0:jct_h, 0:jct_w]

		jctmsk = mk().junction(jct)
		if jctmsk[0][int(jct_w/2)] == 255: jctmsk = mt().reverse_area(jctmsk)

		bot[0:jct_h, 0:jct_w] = cv2.bitwise_and(jct, jct, mask=jctmsk)
		bot_alpha[0:jct_h, 0:jct_w] = cv2.bitwise_and(jct_alpha, jct_alpha, mask=jctmsk)
		#===============  Process junction area  ===============


		#===============  Denoise #2  ===============
		_, mask3 = cv2.threshold(cv2.cvtColor(bot.copy(), cv2.COLOR_RGB2GRAY),1,255,cv2.THRESH_BINARY)
		ed3 = mt().erode_dilate(mask3)
		CdeN = mt().Contour_deNoise(ed3)
		bot = cv2.bitwise_and(bot, bot, mask=CdeN)
		bot_alpha = cv2.bitwise_and(bot_alpha, bot_alpha, mask=CdeN)
		#=============== Denoise #2  ===============


		#===============  Combine #1  ===============
		rsBottom = np.concatenate([bot, bot_alpha.reshape(bot_h+bias, bot_w, 1)], axis=2)
		#plt.imshow(rsBottom, cmap="gray"); plt.show()
		#===============  Combine #1  ===============


		if save_mask:
			print("[Bottom {}]: Save mask as {}-Bottom-mask.jpg".format(getTime(), imgname))
			maskTitles = ["skin", "Bottom #1", "Bottom #2", "skin e&d", "Bottom #1 e&d", "Bottom #2 e&d"]
			smasks = [mask1, mask2, mask3, ed1, ed2, ed3]
			smcnt = 0
			_r_, _c_ = 2,3
			for smc_ in range(_r_*_c_):
				smcnt+=1
				plt.subplot(_r_, _c_, smcnt)
				plt.title(maskTitles[smc_])
				plt.imshow(smasks[smc_], cmap="gray")
				plt.axis("off")
			plt.savefig("cut/"+imgname+"-Bottom-mask.jpg")

		if save_jct:
			print("[Bottom {}]: Save junction as {}-Bottom-junction.jpg".format(getTime(), imgname))
			maskTitles = ["junction", "junction mask"]
			smasks = [jct, jctmsk]
			smcnt = 0
			_r_, _c_ = 1,2
			for smc_ in range(_r_*_c_):
				smcnt+=1
				plt.subplot(_r_, _c_, smcnt)
				plt.title(maskTitles[smc_])
				plt.imshow(smasks[smc_], cmap="gray")
				plt.axis("off")
			plt.savefig("cut/"+imgname+"-Bottom-junction.jpg")

		if save_bottom:
			print("[Bottom {}]: Save bottom as {}-Bottom.png".format(getTime(), imgname))
			misc.imsave("cut/"+imgname+"-Bottom.png", rsBottom)


		#===============  Save point as json  ===============
		print("[Bottom {}] Save points as {}-Bottom-point.json".format(getTime(), imgname))
		js = {}
		ori_img = {}
		ori_img["height"] = int(h); ori_img["width"] = int(w); ori_img["channel"] = int(c)
		js["image"] = ori_img

		bot_js = {}
		bot_js["x1"] = int(bot_x1); bot_js["y1"] = int(bot_y1-bias)
		bot_js["x2"] = int(bot_x2); bot_js["y2"] = int(bot_y2)
		bot_js["height"] = int(bot_h+bias); bot_js["width"] = int(bot_w);
		js["bottom"] = bot_js

		with open("cut/"+imgname+"-Bottom-point.json", "w") as jw:
			json.dump(js, jw)
		#===============  Save point as json  ===============
	else:
		print("[Bottom {}]: {}".format(getTime(), result))
	print("[System {}]: cropBottom Finish, Cost: {:.5f}sec\n".format(getTime(), time.time()-ts))


#if __name__ == '__main__':
#	Bottom(imgname="img_00000207-nobg.png".split(".")[0])