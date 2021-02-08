import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np

class OGDA:
	def __init__(
		self, 
		step_size=0.0001, 
		model=None,
		loss=None,
		dataset=None
	):
		self.model = model
		self.loss = loss
		self.dataset = dataset
		self.step_size = step_size
		self.optimizer = keras.optimizers.SGD(1.)
		self.past_disc_gradient = None
		self.past_gen_gradient = None
		self.name = 'OGDA (' + str(self.step_size) + ')'

	def train_step(self, batch_size):
		real_images = self.dataset.batch(batch_size)

		random_latent_vectors = tf.random.normal(shape=(batch_size, self.model.latent_dim))

		with tf.GradientTape() as tape:
			fake_images = self.model.generator(random_latent_vectors, training=True)

			d_loss = self.loss.disc_loss(real_input=real_images, fake_input=fake_images)

		d_gradient = tape.gradient(d_loss, self.model.discriminator.trainable_variables)
		if self.past_disc_gradient == None:
			grad = [self.step_size * d_gradient[i]
					for i in range(len(d_gradient))]
			self.past_disc_gradient = d_gradient
			self.optimizer.apply_gradients(
				zip(grad, self.model.discriminator.trainable_variables)
			)
		else:
			grad = [2 * self.step_size * d_gradient[i]
					- self.step_size * self.past_disc_gradient[i]
					for i in range(len(d_gradient))]
			self.past_disc_gradient = d_gradient
			self.optimizer.apply_gradients(
				zip(grad, self.model.discriminator.trainable_variables)
			)

		random_latent_vectors = tf.random.normal(shape=(batch_size, self.model.latent_dim))
		with tf.GradientTape() as tape:
			generated_images = self.model.generator(random_latent_vectors, training=True)
			gen_img_logits = self.model.discriminator(generated_images, training=True)

			g_loss = self.loss.gen_loss(gen_img_logits)

		gen_gradient = tape.gradient(g_loss, self.model.generator.trainable_variables)
		if self.past_gen_gradient == None:
			grad = [self.step_size * gen_gradient[i]
					for i in range(len(gen_gradient))]
			self.past_gen_gradient = gen_gradient
			self.optimizer.apply_gradients(
				zip(grad, self.model.generator.trainable_variables)
			)
		else:
			grad = [2 * self.step_size * gen_gradient[i]
					- self.step_size * self.past_gen_gradient[i]
					for i in range(len(gen_gradient))]
			self.past_gen_gradient = gen_gradient
			self.optimizer.apply_gradients(
				zip(grad, self.model.generator.trainable_variables)
			)

		return {"d_loss": K.eval(d_loss), "g_loss": K.eval(g_loss)}