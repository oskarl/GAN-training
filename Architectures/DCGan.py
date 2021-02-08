import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import Adam, SGD
import numpy as np

class DCGan:
	def __init__(self, img_shape=None, latent_dim=None, sigmoid=True):
		self.img_shape = img_shape
		self.channels = img_shape[2]
		self.latent_dim = latent_dim

		self.discriminator = self.build_discriminator(sigmoid=sigmoid)
		self.generator = self.build_generator()
		self.name = "DCGan"

	def build_generator(self):
		model = Sequential()

		model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
		model.add(Reshape((7, 7, 128)))
		model.add(UpSampling2D())
		model.add(Conv2D(128, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
		model.add(UpSampling2D())
		model.add(Conv2D(64, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
		model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
		model.add(Activation("tanh"))

		noise = Input(shape=(self.latent_dim,))
		img = model(noise)

		return Model(noise, img)

	def build_discriminator(self, sigmoid=True):
		model = Sequential()

		model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
		model.add(LeakyReLU(alpha=0.2))
		#model.add(Dropout(0.25))
		model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
		#model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		#model.add(Dropout(0.25))
		model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
		#model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		#model.add(Dropout(0.25))
		model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
		#model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		#model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid' if sigmoid else None))

		model.summary()

		img = Input(shape=self.img_shape)
		validity = model(img)

		return Model(img, validity)