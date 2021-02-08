import tensorflow as tf
import numpy as np
import keras.backend as K
import keras

# remember to set sigmoid = False on discriminator when initiating model

class WGANGP:
	def __init__(self, model, gradient_penalty=10):
		self.gp_weight = gradient_penalty
		self.model = model
		self.name = 'WGAN-GP'

	def disc_loss(self, real_input, fake_input):
		real_logits = self.model.discriminator(real_input, training=True)
		fake_logits = self.model.discriminator(fake_input, training=True)

		real_loss = tf.reduce_mean(real_logits)
		fake_loss = tf.reduce_mean(fake_logits)
		
		gp = self.gradient_penalty(real_input, fake_input)

		return fake_loss - real_loss + gp * self.gp_weight

	def gen_loss(self, disc_output):
		loss = -tf.reduce_mean(disc_output)
		return loss

	def gradient_penalty(self, real_images, fake_images):
		""" Calculates the gradient penalty.

		This loss is calculated on an interpolated image
		and added to the discriminator loss.
		"""
		batch_size = real_images.shape[0]
		# Get the interpolated image
		alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
		diff = fake_images - real_images
		interpolated = real_images + alpha * diff

		with tf.GradientTape() as gp_tape:
		    gp_tape.watch(interpolated)
		    # 1. Get the discriminator output for this interpolated image.
		    pred = self.model.discriminator(interpolated, training=True)

		# 2. Calculate the gradients w.r.t to this interpolated image.
		grads = gp_tape.gradient(pred, [interpolated])[0]
		# 3. Calculate the norm of the gradients.
		norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
		gp = tf.reduce_mean((norm - 1.0) ** 2)
		return gp