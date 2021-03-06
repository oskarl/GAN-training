import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np

class EG:
	def __init__(
		self, 
		step_size=0.0001, 
		same_sample=False,
		model=None,
		loss=None,
		dataset=None
	):
		self.model = model
		self.loss = loss
		self.dataset = dataset
		self.step_size = step_size
		self.same_sample = same_sample
		self.optimizer = keras.optimizers.SGD(step_size)
		self.name = 'EG' + (' Same Sample' if same_sample else '') + ' (' + str(self.step_size) + ')'

	def train_step(self, batch_size):
		real_images = self.dataset.batch(batch_size)

		gen_weights = []
		for layer in self.model.generator.layers:
			gen_weights.append(layer.get_weights())
		disc_weights = []
		for layer in self.model.discriminator.layers:
			disc_weights.append(layer.get_weights())

		# lookahead:

		random_latent_vectors_1 = tf.random.normal(shape=(batch_size, self.model.latent_dim))

		with tf.GradientTape() as tape:
			fake_images = self.model.generator(random_latent_vectors_1, training=True)

			d_loss = self.loss.disc_loss(real_input=real_images, fake_input=fake_images)

		d_gradient = tape.gradient(d_loss, self.model.discriminator.trainable_variables)

		random_latent_vectors_2 = tf.random.normal(shape=(batch_size, self.model.latent_dim))
		with tf.GradientTape() as tape:
			generated_images = self.model.generator(random_latent_vectors_2, training=True)
			gen_img_logits = self.model.discriminator(generated_images, training=True)

			g_loss = self.loss.gen_loss(gen_img_logits)

		gen_gradient = tape.gradient(g_loss, self.model.generator.trainable_variables)
		
		self.optimizer.apply_gradients(
			zip(d_gradient, self.model.discriminator.trainable_variables)
		)
		self.optimizer.apply_gradients(
			zip(gen_gradient, self.model.generator.trainable_variables)
		)

		d_gradient_1 = d_gradient
		gen_gradient_1 = gen_gradient

		# update:

		if not self.same_sample:
			real_images = self.dataset.batch(batch_size)
			random_latent_vectors_1 = tf.random.normal(shape=(batch_size, self.model.latent_dim))
			random_latent_vectors_2 = tf.random.normal(shape=(batch_size, self.model.latent_dim))

		with tf.GradientTape() as tape:
			fake_images = self.model.generator(random_latent_vectors_1, training=True)

			d_loss = self.loss.disc_loss(real_input=real_images, fake_input=fake_images)

		d_gradient = tape.gradient(d_loss, self.model.discriminator.trainable_variables)

		with tf.GradientTape() as tape:
			generated_images = self.model.generator(random_latent_vectors_2, training=True)
			gen_img_logits = self.model.discriminator(generated_images, training=True)

			g_loss = self.loss.gen_loss(gen_img_logits)

		gen_gradient = tape.gradient(g_loss, self.model.generator.trainable_variables)

		#self.model.generator.set_weights(gen_weights)
		#self.model.discriminator.set_weights(disc_weights)
		self.optimizer.apply_gradients(zip(-1 * d_gradient_1, self.model.discriminator.trainable_variables))
		self.optimizer.apply_gradients(zip(-1 * gen_gradient_1, self.model.generator.trainable_variables))

		self.optimizer.apply_gradients(
			zip(d_gradient, self.model.discriminator.trainable_variables)
		)
		self.optimizer.apply_gradients(
			zip(gen_gradient, self.model.generator.trainable_variables)
		)

		return {"d_loss": K.eval(d_loss), "g_loss": K.eval(g_loss)}