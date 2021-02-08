import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import Adam, SGD
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras import layers


def conv_block(
	x,
	filters,
	activation,
	kernel_size=(3, 3),
	strides=(1, 1),
	padding="same",
	use_bias=True,
	use_bn=False,
	use_dropout=False,
	drop_value=0.5,
):
	x = layers.Conv2D(
		filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
	)(x)
	if use_bn:
		x = layers.BatchNormalization()(x)
		x = activation(x)
	if use_dropout:
		x = layers.Dropout(drop_value)(x)
	return x

def upsample_block(
	x,
	filters,
	activation,
	kernel_size=(3, 3),
	strides=(1, 1),
	up_size=(2, 2),
	padding="same",
	output_padding=None,
	use_bn=False,
	use_bias=True,
	use_dropout=False,
	drop_value=0.3,
	use_wn=False,
):
	layer = layers.Conv2DTranspose(
		filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, output_padding=output_padding
	)
	if use_wn:
		x = tfp.layers.weight_norm.WeightNorm(layer)(x)
	else:
		x = layer(x)

	if use_bn:
		x = layers.BatchNormalization()(x)

	if activation:
		x = activation(x)
	if use_dropout:
		x = layers.Dropout(drop_value)(x)
	return x

class OptimisticMirrorDescent2018:
	def __init__(self, img_shape=None, latent_dim=None, sigmoid=True):
		self.img_shape = img_shape
		self.channels = img_shape[2]
		self.latent_dim = latent_dim
		self.sigmoid = sigmoid

		self.discriminator = self.build_discriminator(sigmoid=sigmoid)
		self.generator = self.build_generator()
		self.name = "OMD2018Architecture"

	def build_generator(self):
		noise = layers.Input(shape=(self.latent_dim,))
		x = layers.Dense(4 * 4 * 512, use_bias=False)(noise)
		x = layers.BatchNormalization()(x)
		x = layers.ReLU()(x)

		x = layers.Reshape((4, 4, 512))(x)
		x = upsample_block(
			x,
			256,
			layers.ReLU(),
			kernel_size=(4, 4),
			strides=(2, 2),
			use_bias=True,
			use_bn=True,
			padding="same",
			use_dropout=False,
		)
		x = upsample_block(
			x,
			128,
			layers.ReLU(),
			kernel_size=(4, 4),
			strides=(2, 2),
			use_bias=True,
			use_bn=True,
			padding="same",
			use_dropout=False,
		)
		x = upsample_block(
			x,
			64,
			layers.ReLU(),
			kernel_size=(4, 4),
			strides=(2, 2),
			use_bias=True,
			use_bn=True,
			padding="same",
			use_dropout=False,
		)
		x = upsample_block(
			x, 3, layers.Activation("tanh"), kernel_size=(4, 4), strides=(1, 1), use_bias=True, use_bn=False, use_wn=True
		)
		# At this point, we have an output which has the same shape as the input, (32, 32, 1).
		# We will use a Cropping2D layer to make it (28, 28, 1).
		#x = layers.Cropping2D((2, 2))(x)

		g_model = keras.models.Model(noise, x, name="generator")
		return g_model

	def build_discriminator(self, sigmoid=True):
		img_input = layers.Input(shape=self.img_shape)
		# Zero pad the input to make the input images size to (32, 32, 1).
		#x = layers.ZeroPadding2D((2, 2))(img_input)
		x = conv_block(
			img_input,
			64,
			kernel_size=(3, 3),
			strides=(1, 1),
			use_bn=False,
			use_bias=True,
			activation=layers.LeakyReLU(0.2),
			use_dropout=False,
			drop_value=0.3,
		)
		x = conv_block(
			x,
			128,
			kernel_size=(3, 3),
			strides=(2, 2),
			use_bn=False,
			activation=layers.LeakyReLU(0.2),
			use_bias=True,
			use_dropout=False,
			drop_value=0.3,
		)
		x = conv_block(
			x,
			128,
			kernel_size=(3, 3),
			strides=(1, 1),
			use_bn=False,
			activation=layers.LeakyReLU(0.2),
			use_bias=True,
			use_dropout=False,
			drop_value=0.3,
		)
		x = conv_block(
			x,
			256,
			kernel_size=(3, 3),
			strides=(2, 2),
			use_bn=False,
			activation=layers.LeakyReLU(0.2),
			use_bias=True,
			use_dropout=False,
			drop_value=0.3,
		)
		x = conv_block(
			x,
			256,
			kernel_size=(3, 3),
			strides=(1, 1),
			use_bn=False,
			activation=layers.LeakyReLU(0.2),
			use_bias=True,
			use_dropout=False,
			drop_value=0.3,
		)
		x = conv_block(
			x,
			512,
			kernel_size=(3, 3),
			strides=(2, 2),
			use_bn=False,
			activation=layers.LeakyReLU(0.2),
			use_bias=True,
			use_dropout=False,
			drop_value=0.3,
		)
		x = conv_block(
			x,
			512,
			kernel_size=(3, 3),
			strides=(1, 1),
			use_bn=False,
			activation=layers.LeakyReLU(0.2),
			use_bias=True,
			use_dropout=False,
			drop_value=0.3,
		)

		x = layers.Flatten()(x)
		#x = layers.Dropout(0.2)(x)
		x = layers.Dense(1, activation='sigmoid' if sigmoid else None)(x)

		d_model = keras.models.Model(img_input, x, name="discriminator")
		return d_model