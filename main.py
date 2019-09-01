import warnings, os
warnings.simplefilter("ignore")

from keras.models import load_model
from scipy import misc
import cv2, numpy as np
from Utils import reverse, normalize
import Utils

Utils.SET_GPU_MEM() #set GPU memory limit

#   =============  read row image  ================
rowImgname = "yourIamgeName.jpg".split(".")[0]
rowImg = misc.imread(rowImgname+".jpg")
h,w,c = rowImg.shape
os.makedirs(rowImgname, exist_ok=True)


# ======================================================#
#														#
#					Remove background					#
#														#
# ======================================================#

#   =============  generative mask_nobg  ================
model_nobg = load_model("models/yourModelName.h5")
fake_nobg = cv2.resize(reverse(model_nobg.predict(normalize(rowImg))), (w,h))
Utils.CLEAR_SESSION()
del model_nobg

#   =============  clean mask_nobg  ================
mask_nobg = cv2.medianBlur(Utils.noisetool().deNoise1_er(Utils.masktool().binary(fake_nobg, 32)), 7)

#   =============  bitwise and  ================
nobg_rgb = cv2.bitwise_and(rowImg, rowImg, mask=mask_nobg)
nobg_final = np.concatenate([nobg_rgb, mask_nobg.reshape(h,w,1)], axis=-1)

misc.imsave(rowImgname+"/"+rowImgname+"-gan.jpg", fake_nobg)
misc.imsave(rowImgname+"/"+rowImgname+"-mask.jpg", mask_nobg)
misc.imsave(rowImgname+"/"+rowImgname+"-nobg.png", nobg_final)
