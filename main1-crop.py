import warnings, os
warnings.simplefilter("ignore")

from keras.models import load_model
from scipy import misc
import cv2, numpy as np, json
from utils import reverse, normalize
import utils

utils.SET_GPU_MEM() #set GPU memory limit

#   =============  read row image  ================
rowImgname = "sample1.jpg".split(".")[0]
rowImg = misc.imread(rowImgname+".jpg")
h,w,c = rowImg.shape
os.makedirs(rowImgname, exist_ok=True)


# ======================================================#
#														#
#					1. Remove background				#
#														#
# ======================================================#

#   =============  1.1. generative mask_nobg  ================
model_nobg = load_model("models/Removebg_mask_250.h5")
fake_nobg = cv2.resize(reverse(model_nobg.predict(normalize(rowImg))), (w,h))
utils.CLEAR_SESSION()
del model_nobg

#   =============  1.2. clean mask_nobg  ================
mask_nobg = cv2.medianBlur(utils.noisetool().deNoise1_er(utils.masktool().binary(fake_nobg, 32)), 7)

#   =============  1.3. bitwise and  ================
nobg_rgb = cv2.bitwise_and(rowImg, rowImg, mask=mask_nobg)
nobg_final = np.concatenate([nobg_rgb, mask_nobg.reshape(h,w,1)], axis=-1)

misc.imsave(rowImgname+"/"+rowImgname+"-gan.jpg", fake_nobg)
misc.imsave(rowImgname+"/"+rowImgname+"-mask.jpg", mask_nobg)
misc.imsave(rowImgname+"/"+rowImgname+"-nobg.png", nobg_final)


# ======================================================#
#														#
#					2. Crop Head and Arms				#
#														#
# ======================================================#

pngImg = misc.imread(rowImgname+"/"+rowImgname+"-nobg.png")
alpha = pngImg[...,-1]

#   =============  2.1. generative mask_CpHA  ================
model_CpHA = load_model("models/CropHA_mask.h5")
fake_CpHA = cv2.resize(reverse(model_CpHA.predict(normalize(pngImg))), (w,h))
utils.CLEAR_SESSION()
del model_CpHA

fake_head = fake_CpHA[...,0]
fake_armL = fake_CpHA[...,1]
fake_armR = fake_CpHA[...,2]

misc.imsave(rowImgname+"/"+rowImgname+"-head-gan.jpg", fake_head)
misc.imsave(rowImgname+"/"+rowImgname+"-armL-gan.jpg", fake_armL)
misc.imsave(rowImgname+"/"+rowImgname+"-armR-gan.png", fake_armR)

#   =============  2.2. clean mask_nobg  ================
mask_head = cv2.medianBlur(utils.masktool().binary(fake_head, 32), 7)
mask_armL = cv2.medianBlur(utils.masktool().binary(fake_armL, 32), 7)
mask_armR = cv2.medianBlur(utils.masktool().binary(fake_armR, 32), 7)

misc.imsave(rowImgname+"/"+rowImgname+"-head-mask.jpg", mask_head)
misc.imsave(rowImgname+"/"+rowImgname+"-armL-mask.jpg", mask_armL)
misc.imsave(rowImgname+"/"+rowImgname+"-armR-mask.png", mask_armR)

#   =============  2.3. bitwise and  ================
head_rgb = cv2.bitwise_and(pngImg, pngImg, mask=mask_head)
misc.imsave(rowImgname+"/"+rowImgname+"-head.png", head_rgb)

armL_rgb = cv2.bitwise_and(pngImg, pngImg, mask=mask_armL)
misc.imsave(rowImgname+"/"+rowImgname+"-armL.png", armL_rgb)

armR_rgb = cv2.bitwise_and(pngImg, pngImg, mask=mask_armR)
misc.imsave(rowImgname+"/"+rowImgname+"-armR.png", armR_rgb)


# ======================================================#
#														#
#					3. Crop Clothes						#
#														#
# ======================================================#

#   =============  3.1. Open YOLO model  ================
options = {"pbLoad": "models/yolo-top.pb", "metaLoad": "models/yolo-top.meta", "json": True}
yolo = utils.yolonet(options)
model_yolo = yolo.createNet()
rs = model_yolo.return_predict(cv2.imread(rowImgname+".jpg"))
pure = pngImg - head_rgb - armL_rgb - armR_rgb

#   =============  3.2. Crop top  ================
ps_top = yolo.parsing(rs, keys="top")
result_top = utils.top(pure, ps_top)
try:
	misc.imsave(rowImgname+"/"+rowImgname+"-top0-crop.png", result_top[0])
	#misc.imsave(rowImgname+"/"+rowImgname+"-top1-skinMask.png", result_top[1])
	#misc.imsave(rowImgname+"/"+rowImgname+"-top2-skinMask_ed.png", result_top[2])
	misc.imsave(rowImgname+"/"+rowImgname+"-top1-topMask.png", result_top[1])
	misc.imsave(rowImgname+"/"+rowImgname+"-top2-topMask_ed.png", result_top[2])
	misc.imsave(rowImgname+"/"+rowImgname+"-top3-area.png", result_top[3])
	misc.imsave(rowImgname+"/"+rowImgname+"-top.png", result_top[4])
	with open(rowImgname+"/"+rowImgname+"-top.json", "w") as jwtop:
		json.dump(result_top[-1], jwtop)
except:
	pass

#   =============  3.3. Crop bottom  ================
ps_bot = yolo.parsing(rs, keys="bottom")
result_bot = utils.bottom(pure, ps_bot)
try:
	misc.imsave(rowImgname+"/"+rowImgname+"-bot0-crop.png", result_bot[0])
	#misc.imsave(rowImgname+"/"+rowImgname+"-bot1-botMask.png", result_bot[1])
	misc.imsave(rowImgname+"/"+rowImgname+"-bot1-botMask_noJCT.png", result_bot[1])
	#misc.imsave(rowImgname+"/"+rowImgname+"-bot1-botMask_ed.png", result_bot[3])
	misc.imsave(rowImgname+"/"+rowImgname+"-bot2-botMask_noJCT_ed.png", result_bot[2])
	misc.imsave(rowImgname+"/"+rowImgname+"-bot3-junction.png", result_bot[3])
	misc.imsave(rowImgname+"/"+rowImgname+"-bot4-mask_junction.png", result_bot[4])
	misc.imsave(rowImgname+"/"+rowImgname+"-bot5-area.png", result_bot[5])
	misc.imsave(rowImgname+"/"+rowImgname+"-bottom.png", result_bot[6])
	with open(rowImgname+"/"+rowImgname+"-bottom.json", "w") as jwbot:
		json.dump(result_bot[-1], jwbot)
except:
	pass