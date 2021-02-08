import tensorflow as tf
import numpy as np
import keras.backend as K
import keras

class BCE:
	def __init__(self, model):
		self.bce = keras.losses.BinaryCrossentropy(from_logits=False)
		self.model = model
		self.name = 'BinaryCrossentropy'

	def disc_loss(self, real_input, fake_input):
		real_logits = self.model.discriminator(real_input, training=True)
		fake_logits = self.model.discriminator(fake_input, training=True)

		r_loss = self.bce(y_true=tf.ones(real_logits.shape), y_pred=real_logits)
		f_loss = self.bce(y_true=tf.zeros(fake_logits.shape), y_pred=fake_logits)
		return 0.5 * (K.mean(r_loss) + K.mean(f_loss))

	def gen_loss(self, disc_output):
		loss = self.bce(y_true=tf.ones(disc_output.shape), y_pred=disc_output)
		return K.mean(loss)