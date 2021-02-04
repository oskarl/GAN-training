import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import Adam, SGD
import numpy as np

class DenseNetwork:
	def __init__(self, img_shape=None, latent_dim=None):
		self.img_shape = img_shape
		self.latent_dim = latent_dim

		self.discriminator = self.build_discriminator()
		self.generator = self.build_generator()
		self.name = "DenseNetwork"

	def build_generator(self):
		model = Sequential()

		model.add(Dense(256, input_dim=self.latent_dim))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(1024))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(np.prod(self.img_shape), activation='tanh'))
		model.add(Reshape(self.img_shape))

		#model.summary()

		noise = Input(shape=(self.latent_dim,))
		img = model(noise)

		return Model(noise, img)

	def build_discriminator(self):
		model = Sequential()

		model.add(Flatten(input_shape=self.img_shape))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(256))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(1, activation='sigmoid'))
		#model.summary()

		img = Input(shape=self.img_shape)
		validity = model(img)

		return Model(img, validity)