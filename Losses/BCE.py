import tensorflow as tf
import numpy as np
import keras.backend as K
import keras

class BCE:
	def __init__(self):
		self.bce = keras.losses.BinaryCrossentropy(from_logits=False)
		self.name = 'BinaryCrossentropy'

	def disc_loss(self, output, fake):
		loss = self.bce(y_true=tf.zeros(output.shape) if fake else tf.ones(output.shape), y_pred=output)
		return K.mean(loss)

	def gen_loss(self, disc_output):
		return self.disc_loss(disc_output, False)