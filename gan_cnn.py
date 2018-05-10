from keras.layers import Input,Dense,Activation,Dropout
from keras.layers import BatchNormalization,ZeroPadding2D
from keras.layers import Reshape,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D,UpSampling2D
from keras.models import Sequential,Model
from keras.optimizers import Adam

import cv2
import numpy as np
from batching import get_from_dataset,normalize,get_from_path


class DCGAN():
	def __init__(self):
		self.img_length = 64
		self.img_width = 64
		self.num_channels = 3
		self.img_shape = (64,64,3)
		self.noise_shape = (100,)
		
		self.batch_size = 64
		self.num_epochs = 30000

		self.generator = self.getGenerator()
		self.discriminator = self.getDiscriminator()

		opt = Adam(0.0002,0.5)
		
		self.discriminator.compile(
			loss='binary_crossentropy',
			optimizer=opt,
			metrics=['accuracy'])

		self.generator.compile(
			loss='binary_crossentropy',
			optimizer=opt)

		noise = Input(shape=self.noise_shape)
		img = self.generator(noise)

		self.discriminator.trainable = False
		validity = self.discriminator(img)

		self.combined = Model(noise,validity)
		self.combined.compile(
			loss='binary_crossentropy',
			optimizer=opt)


	def getDiscriminator(self):

		model = Sequential()

		model.add(Conv2D(32,3,strides=2,padding='same',input_shape=self.img_shape))
		model.add(LeakyReLU(0.2))
		model.add(Dropout(0.25))

		model.add(Conv2D(64,3,strides=2,padding='same'))
		model.add(ZeroPadding2D(padding=((0,1),(0,1))))
		model.add(LeakyReLU(0.2))
		model.add(Dropout(0.25))
		model.add(BatchNormalization(momentum=0.8))

		model.add(Conv2D(128,3,strides=2,padding='same'))
		model.add(LeakyReLU(0.2))
		model.add(Dropout(0.25))
		model.add(BatchNormalization(momentum=0.8))

		model.add(Conv2D(256,3,strides=2,padding='same'))
		model.add(LeakyReLU(0.2))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(1,activation='sigmoid'))

		img = Input(shape=self.img_shape)
		validity = model(img)
		model = Model(img,validity)

		return model

	def getGenerator(self):

		model = Sequential()

		model.add(Dense(128*7*7,activation='relu',input_shape=self.noise_shape))
		model.add(Reshape((7,7,128)))
		model.add(BatchNormalization(momentum=0.8))
		model.add(UpSampling2D())

		model.add(Conv2D(128,3,padding='same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(momentum=0.8))
		model.add(UpSampling2D())
		
		model.add(Conv2D(64,3,padding='same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(momentum=0.8))
		model.add(UpSampling2D())
		
		model.add(Conv2D(self.num_channels,3,padding='same'))
		model.add(Activation('tanh'))

		noise = Input(shape=self.noise_shape)
		img = model(noise)
		model = Model(noise,img)

		return model


	def train(self):

		half_batch = self.batch_size//2
		noise_shape = self.noise_shape

		for epoch in range(self.num_epochs):
			
			noise = np.random.normal(0,1,(half_batch,100))
			gen_imgs = self.generator.predict(noise)

			imgs = get_from_path('data/all/dogs2/*.jpg')
			real_loss = self.discriminator.train_on_batch(imgs,np.ones((half_batch,1)))
			fake_loss = self.discriminator.train_on_batch(gen_imgs,np.zeros((half_batch,1)))
			disc_loss = 0.5*np.add(real_loss,fake_loss)

			noise = np.random.normal(0,1,(batch_size,100))
			gen_loss = self.generator.train_on_batch(noise,np.ones((batch_size,1)))

			if epoch%200==0:
				self.sample_images(epoch)
				print epoch,disc_loss[0],100*disc_loss[1],gen_loss


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

d = DCGAN()
d.train()