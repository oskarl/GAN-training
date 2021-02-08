import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np

class SGDA:
	def __init__(
		self, 
		step_size=0.02, 
		model=None,
		loss=None,
		dataset=None
	):
		self.model = model
		self.loss = loss
		self.dataset = dataset
		self.step_size = step_size
		self.d_optimizer = keras.optimizers.SGD(step_size)
		self.g_optimizer = keras.optimizers.SGD(step_size)
		self.name = 'SGD (' + str(self.step_size) + ')'

	def train_step(self, batch_size):
		real_images = self.dataset.batch(batch_size)

		random_latent_vectors = tf.random.normal(shape=(batch_size, self.model.latent_dim))

		with tf.GradientTape() as tape:
			fake_images = self.model.generator(random_latent_vectors, training=True)

			d_loss = self.loss.disc_loss(real_input=real_images, fake_input=fake_images)
		
		d_gradient = tape.gradient(d_loss, self.model.discriminator.trainable_variables)
		self.d_optimizer.apply_gradients(
			zip(d_gradient, self.model.discriminator.trainable_variables)
		)
		
		random_latent_vectors = tf.random.normal(shape=(batch_size, self.model.latent_dim))
		with tf.GradientTape() as tape:
			generated_images = self.model.generator(random_latent_vectors, training=True)
			gen_img_logits = self.model.discriminator(generated_images, training=True)

			g_loss = self.loss.gen_loss(gen_img_logits)

		gen_gradient = tape.gradient(g_loss, self.model.generator.trainable_variables)
		self.g_optimizer.apply_gradients(
			zip(gen_gradient, self.model.generator.trainable_variables)
		)
		
		return {"d_loss": K.eval(d_loss), "g_loss": K.eval(g_loss)}