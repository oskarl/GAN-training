import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np

class OptimisticAdam:
	def __init__(
		self, 
		step_size=0.0001,
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
		self.beta1 = beta1
		self.beta2 = beta2
		self.t = 0
		self.eps = 1e-7
		self.gen_m = None
		self.gen_v = None
		self.d_m = None
		self.d_v = None

		self.optimizer = keras.optimizers.SGD(1.)
		self.name = 'OptimisticAdam (' + str(self.step_size) + ')'

	def train_step(self, batch_size):
		real_images = self.dataset.batch(batch_size)

		random_latent_vectors = tf.random.normal(shape=(batch_size, self.model.latent_dim))

		self.t += 1

		with tf.GradientTape() as tape:
			fake_images = self.model.generator(random_latent_vectors, training=True)

			d_loss = self.loss.disc_loss(real_input=real_images, fake_input=fake_images)

		d_gradient = tape.gradient(d_loss, self.model.discriminator.trainable_variables)
		
		grad = None
		if self.d_m == None:
			self.d_m = [(1-self.beta1) * g for g in d_gradient]
			self.d_v = [(1-self.beta2) * g * g for g in d_gradient]
			mth = [m / (1-self.beta1**self.t) for m in self.d_m]
			vth = [v / (1-self.beta2**self.t) for v in self.d_v]

			grad = [self.step_size * mth[i] / (tf.sqrt(vth[i]) + self.eps)
					for i in range(len(d_gradient))]
		else:
			mthm1 = [m / (1-self.beta1**(self.t-1)) for m in self.d_m]
			vthm1 = [v / (1-self.beta2**(self.t-1)) for v in self.d_v]
			self.d_m = [self.beta1 * self.d_m[i] + (1-self.beta1) * d_gradient[i] for i in range(len(d_gradient))]
			self.d_v = [self.beta2 * self.d_v[i] + (1-self.beta2) * d_gradient[i] * d_gradient[i] for i in range(len(d_gradient))]
			mth = [m / (1-self.beta1**self.t) for m in self.d_m]
			vth = [v / (1-self.beta2**self.t) for v in self.d_v]

			grad = [2 * self.step_size * mth[i] / (tf.sqrt(vth[i]) + self.eps)
					- self.step_size * mthm1[i] / (tf.sqrt(vthm1[i]) + self.eps)
					for i in range(len(d_gradient))]
		self.optimizer.apply_gradients(
			zip(grad, self.model.discriminator.trainable_variables)
		)

		random_latent_vectors = tf.random.normal(shape=(batch_size, self.model.latent_dim))
		with tf.GradientTape() as tape:
			generated_images = self.model.generator(random_latent_vectors, training=True)
			gen_img_logits = self.model.discriminator(generated_images, training=True)

			g_loss = self.loss.gen_loss(gen_img_logits)

		gen_gradient = tape.gradient(g_loss, self.model.generator.trainable_variables)
		
		grad = None
		if self.gen_m == None:
			self.gen_m = [(1-self.beta1) * g for g in gen_gradient]
			self.gen_v = [(1-self.beta2) * g * g for g in gen_gradient]
			mth = [m / (1-self.beta1**self.t) for m in self.gen_m]
			vth = [v / (1-self.beta2**self.t) for v in self.gen_v]

			grad = [self.step_size * mth[i] / (tf.sqrt(vth[i]) + self.eps)
					for i in range(len(gen_gradient))]
		else:
			mthm1 = [m / (1-self.beta1**(self.t-1)) for m in self.gen_m]
			vthm1 = [v / (1-self.beta2**(self.t-1)) for v in self.gen_v]
			self.gen_m = [self.beta1 * self.gen_m[i] + (1-self.beta1) * gen_gradient[i] for i in range(len(gen_gradient))]
			self.gen_v = [self.beta2 * self.gen_v[i] + (1-self.beta2) * gen_gradient[i] * gen_gradient[i] for i in range(len(gen_gradient))]
			mth = [m / (1-self.beta1**self.t) for m in self.gen_m]
			vth = [v / (1-self.beta2**self.t) for v in self.gen_v]

			grad = [2 * self.step_size * mth[i] / (tf.sqrt(vth[i]) + self.eps)
					- self.step_size * mthm1[i] / (tf.sqrt(vthm1[i]) + self.eps)
					for i in range(len(gen_gradient))]

		self.optimizer.apply_gradients(
			zip(grad, self.model.generator.trainable_variables)
		)

		return {"d_loss": K.eval(d_loss), "g_loss": K.eval(g_loss)}