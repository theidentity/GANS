import numpy as np
import cv2


import keras
from keras.models import Sequential,load_model,save_model,Model
from keras.layers import Input,Dense,Reshape,Flatten,Dropout,Activation
from keras.layers import BatchNormalization,ZeroPadding2D
from keras.layers import UpSampling2D,Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator
from batching import get_from_dataset,normalize,get_from_path
from matplotlib import pyplot as plt


img_rows,img_cols = 64,64
inp_shape = (64*64,)
img_shape = (img_rows,img_cols,3)
# img_shape = (img_rows,img_cols,1)
noise_shape = (100,)

num_epochs = 60000
batch_size = 64
half_batch = batch_size//2

train_dir = 'data/all/'
# from keras.datasets import fashion_mnist

class GAN_MLP(object):
	"""docstring for GAN_MLP"""
	def __init__(self):
		self.generator = self.get_generator()
		self.discriminator = self.get_discriminator()
		self.combined = self.combine()
		

	def get_generator(self):

		model = Sequential()
		model.add(Dense(256, input_shape=noise_shape))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(1024))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(np.prod(img_shape), activation='tanh'))
		model.add(Reshape(img_shape))


		noise = Input(shape=noise_shape)
		out_img = model(noise)
		generator = Model(noise,out_img)
		return generator

	def get_discriminator(self):

		model = Sequential()

		model.add(Flatten(input_shape=img_shape))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(256))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(1, activation='sigmoid'))

		in_img = Input(img_shape)
		validity = model(in_img)
		discriminator = Model(in_img,validity)
		return discriminator

	def combine(self):
		# opt = SGD(0.0002,0.5)
		opt = Adam(0.0002,0.5)

		self.generator.compile(
			optimizer=opt,
			loss='binary_crossentropy'
			)

		self.discriminator.compile(
			optimizer=opt,
			loss='binary_crossentropy',
			metrics=['accuracy'])


		inp = Input(shape=(noise_shape))
		out_img = self.generator(inp)
		
		self.discriminator.trainable = False
		validity = self.discriminator(out_img)

		model = Model(inp,validity)
		model.compile(
			optimizer=opt,
			loss='binary_crossentropy')

		return model

	def train(self):

		# (trX,_),(_,_) = fashion_mnist.load_data()
		# trX = normalize(trX,-1,1)

		for epoch in range(num_epochs):

			noise = np.random.normal(0,1,(half_batch,noise_shape[0]))
			gen_imgs = self.generator.predict(noise)
			
			imgs = get_from_path('data/all/dogs2/*.jpg')
			# imgs = get_from_dataset(trX,half_batch)
			# imgs = np.expand_dims(imgs,axis=3)

			real_loss = self.discriminator.train_on_batch(x=imgs,y=np.ones((half_batch,1)))
			fake_loss = self.discriminator.train_on_batch(x=gen_imgs,y=np.zeros((half_batch,1)))
			disc_loss = 0.5*np.add(real_loss,fake_loss)

			noise = np.random.normal(0,1,(batch_size,noise_shape[0]))
			valid_y = np.ones(batch_size,dtype=int)
			gen_loss = self.combined.train_on_batch(x=noise,y=valid_y)

			if epoch%200 == 0:
				print epoch,disc_loss[0],100*disc_loss[1],gen_loss
				self.sample_images(epoch)

	def sample_images(self,epoch):
		r,c = 5,5
		noise = np.random.normal(0,1,(r*c,noise_shape[0]))
		gen_imgs = self.generator.predict(noise)
		gen_imgs = gen_imgs*.5+.5
		gen_imgs = gen_imgs*255
		gen_imgs = np.split(gen_imgs,gen_imgs.shape[0],axis=0)

		out = np.hstack([img[0] for img in gen_imgs])
		name = ''.join(['tmp/',str(epoch),'.png'])
		cv2.imwrite(name,out)

gan = GAN_MLP()
gan.train()