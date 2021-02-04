import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np
from keras.layers import Input
from keras.models import Model

class Adam:
	def __init__(
		self, 
		step_size=0.0002, 
		beta1=0.5,
		beta2=0.999,
		model=None,
		loss=None,
		dataset=None
	):
		self.model = model
		self.loss = loss
		self.dataset = dataset
		self.step_size = step_size
		self.d_optimizer = keras.optimizers.Adam(learning_rate=step_size, beta_1=beta1, beta_2=beta2)
		self.g_optimizer = keras.optimizers.Adam(learning_rate=step_size, beta_1=beta1, beta_2=beta2)
		self.name = 'Adam (' + str(self.step_size) + ', ' + str(beta1) + ', ' + str(beta2) + ')'

	def train_step(self, batch_size):
		real_images = self.dataset.batch(batch_size)
		with tf.GradientTape() as tape:
			real_logits = self.model.discriminator(real_images, training=True)

			d_loss_real = self.loss.disc_loss(output=real_logits, fake=False)
		
		d_gradient = tape.gradient(d_loss_real, self.model.discriminator.trainable_variables)
		self.d_optimizer.apply_gradients(
			zip(d_gradient, self.model.discriminator.trainable_variables)
		)
		
		random_latent_vectors = tf.random.normal(shape=(batch_size, self.model.latent_dim))

		with tf.GradientTape() as tape:
			fake_images = self.model.generator(random_latent_vectors, training=True)

			fake_logits = self.model.discriminator(fake_images, training=True)

			d_loss_fake = self.loss.disc_loss(output=fake_logits, fake=True)

		
		d_gradient = tape.gradient(d_loss_fake, self.model.discriminator.trainable_variables)
		self.d_optimizer.apply_gradients(
			zip(d_gradient, self.model.discriminator.trainable_variables)
		)

		d_loss = 0.5 * (d_loss_real + d_loss_fake)
		
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