import os, numpy as np

class DataTools():
	def load_data(self, dirs, dataset_name):
		self.dirs = dirs
		self.dataset_name = dataset_name+".npz"
		print("Load:", os.path.join(self.dirs, self.dataset_name))
		dataset = np.load(os.path.join(self.dirs, self.dataset_name))
		ROW = dataset["ROW"]
		CDT = dataset["CDT"]

		return [ROW, CDT]

	def get_batch(self, data, batch_size):
		imgs_A, imgs_B = data
		randlist = np.random.randint(0, imgs_A.shape[0], size=batch_size)
		imgs_Alist = []
		imgs_Blist = []

		for i in randlist:
			imgs_Alist.append(imgs_A[i])
			imgs_Blist.append(imgs_B[i])

		imgs_A_batch = np.asarray(imgs_Alist)
		imgs_B_batch = np.asarray(imgs_Blist)

		#Regularization imgs -1 to 1
		imgs_A_batch = imgs_A_batch/127.5 - 1
		imgs_B_batch = imgs_B_batch/127.5 - 1

		return imgs_A_batch, imgs_B_batch