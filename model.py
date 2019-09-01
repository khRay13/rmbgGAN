import warnings
warnings.simplefilter('ignore')

from keras.layers import Input, Dropout, Concatenate, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import os, sys
import datetime, cv2, numpy as np

class Utils():
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

class Pix2Pix():
	def __init__(self, dataA, dataB):
		# Input shape
		self.img_rows = self.img_cols = 256
		self.mask_channels = 1
		self.mask_shape = (self.img_rows, self.img_cols, self.mask_channels)
		self.jpg_channels = 3
		self.jpg_shape = (self.img_rows, self.img_cols, self.jpg_channels)

		# Configure data loader
		self.real = dataA
		self.paints = dataB

		# Calculate output shape of D (PatchGAN)
		patch = int(self.img_rows / 2**4)
		self.disc_patch = (patch, patch, 1)

		# Number of filters in the first layer of G and D
		self.gf = 64
		self.df = 64

		# Set optimizer
		self.optimizer = Adam(0.0002, 0.5)

	def build_generator(self):
		"""U-Net Generator"""

		def conv2d(layer_input, filters, f_size=4, bn=True):
			"""Layers used during downsampling"""
			d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
			d = LeakyReLU(alpha=0.2)(d)
			if bn:
				d = BatchNormalization(momentum=0.8)(d)
			return d

		def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
			"""Layers used during upsampling"""
			u = UpSampling2D(size=2)(layer_input)
			u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
			if dropout_rate:
				u = Dropout(dropout_rate)(u)
			u = BatchNormalization(momentum=0.8)(u)
			u = Concatenate()([u, skip_input])
			return u

		# Image input
		d0 = Input(shape=self.jpg_shape)

		# Downsampling
		d1 = conv2d(d0, self.gf, bn=False)
		d2 = conv2d(d1, self.gf*2)
		d3 = conv2d(d2, self.gf*4)
		d4 = conv2d(d3, self.gf*8)
		d5 = conv2d(d4, self.gf*8)
		d6 = conv2d(d5, self.gf*8)
		d7 = conv2d(d6, self.gf*8)

		# Upsampling
		u1 = deconv2d(d7, d6, self.gf*8)
		u2 = deconv2d(u1, d5, self.gf*8)
		u3 = deconv2d(u2, d4, self.gf*8)
		u4 = deconv2d(u3, d3, self.gf*4)
		u5 = deconv2d(u4, d2, self.gf*2)
		u6 = deconv2d(u5, d1, self.gf)

		u7 = UpSampling2D(size=2)(u6)
		out = Conv2D(self.mask_channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
		gen_model = Model(d0, out, name="Generator")

		return gen_model

	def build_discriminator(self):

		def d_layer(layer_input, filters, f_size=4, bn=True):
			"""Discriminator layer"""
			d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
			d = LeakyReLU(alpha=0.2)(d)
			if bn:
				d = BatchNormalization(momentum=0.8)(d)
			return d

		img_A = Input(shape=self.mask_shape)
		img_B = Input(shape=self.jpg_shape)

		# Concatenate image and conditioning image by channels to produce input
		combined_imgs = Concatenate(axis=-1)([img_A, img_B])

		d1 = d_layer(combined_imgs, self.df, bn=False)
		d2 = d_layer(d1, self.df*2)
		d3 = d_layer(d2, self.df*4)
		d4 = d_layer(d3, self.df*8)

		validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

		dis_model = Model([img_A, img_B], validity, name="Discriminator")
		dis_model.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])

		return dis_model

	def build_gan(self, gen, dis):
		# Input images and their conditioning images
		img_A = Input(shape=self.mask_shape)
		img_B = Input(shape=self.jpg_shape)

		# By conditioning on B generate a fake version of A
		fake_A = gen(img_B)

		# For the combined model we will only train the generator
		dis.trainable = False

		# Discriminators determines validity of translated images / condition pairs
		valid = dis([fake_A, img_B])

		gan_model = Model(inputs=[img_A, img_B], outputs=[valid, fake_A], name="p2pGAN")
		gan_model.compile(loss=['mse', 'mae'],
							  loss_weights=[1, 100],
							  optimizer=self.optimizer)
		return gan_model


	def train(self, name, models, epochs, batch_size=1, sample_interval=50):

		start_time = datetime.datetime.now()
		gen, dis, p2pgan = models

		# Adversarial loss ground truths
		valid = np.ones((batch_size,) + self.disc_patch)
		fake = np.zeros((batch_size,) + self.disc_patch)

		batch_count = int(self.real.shape[0]/batch_size)

		for epoch in range(epochs):
			print(" ")
			for bt in range(1, batch_count+1):
				imgs_A, imgs_B = Utils().get_batch([self.real, self.paints], batch_size)
				# ---------------------
				#  Train Discriminator
				# ---------------------

				# Condition on B and generate a translated version
				fake_A = gen.predict(imgs_B)

				# Train the discriminators (original images = real / generated = Fake)
				d_loss_real = dis.train_on_batch([imgs_A, imgs_B], valid)
				d_loss_fake = dis.train_on_batch([fake_A, imgs_B], fake)
				d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

				# -----------------
				#  Train Generator
				# -----------------

				# Train the generators
				g_loss = p2pgan.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

				elapsed_time = datetime.datetime.now() - start_time
				# Plot the progress
				s1 = ("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
																		bt, batch_count,
																		d_loss[0], 100*d_loss[1],
																		g_loss[0],
																		elapsed_time))
				print(s1, end="")

				# If at save interval => save generated image samples
				if bt % sample_interval == 0 or bt == 1:
					pred_img_idx = np.random.randint(0, self.real.shape[0])
					pred_img_A = self.real[pred_img_idx]
					pred_img_B = self.paints[pred_img_idx]
					self.sample_images(name=name, gen=gen, epoch=epoch, batch=bt, pred_img=[pred_img_A, pred_img_B])

			if epoch+1 % 50 == 0:
				self.saveModel(epoch, gen)

	def sample_images(self, name, gen, epoch, batch, pred_img):
		def revert(img):
			return np.uint8(img*127.5+127.5)

		os.makedirs('output-nobgMask', exist_ok=True)
		r, c = 1, 3

		imgs_A, imgs_B = pred_img
		fake_A = gen.predict((imgs_B/127.5-1).reshape(1, self.img_rows, self.img_cols, self.jpg_channels))

		gen_imgs = [cv2.cvtColor(imgs_A[...,0], cv2.COLOR_GRAY2RGB), imgs_B, cv2.cvtColor(revert(fake_A[0]), cv2.COLOR_GRAY2RGB)]

		titles = ['Condition', 'Original', 'Generated']
		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				#print(gen_imgs[cnt].shape)
				plt.subplot(r, c, j+1)
				plt.imshow(gen_imgs[cnt])
				plt.title(titles[cnt])
				plt.axis("off")
				cnt+=1
		fig.savefig("output-nobgMask/{}_{}_{}.png".format(name, epoch, batch))
		plt.close()

	def saveModel(self, epoch, model):
		model.save("Removebg_mask_{}.h5".format(epoch))