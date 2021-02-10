import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np

class HeavyBallOpt(keras.optimizers.Optimizer):
	def __init__(self, learning_rate=0.02, momentum_weight=0.04, name="HeavyBall", **kwargs):
		"""Call super().__init__() and use _set_hyper() to store hyperparameters"""
		super().__init__(name, **kwargs)
		self._set_hyper('momentum_weight', K.variable(momentum_weight, dtype='float32', name='momentum_weight'))
		self._set_hyper("learning_rate", kwargs.get("lr", K.variable(learning_rate, dtype='float32', name='learning_rate'))) # handle lr=learning_rate

	def _create_slots(self, var_list):
		"""For each model variable, create the optimizer variable associated with it.
		TensorFlow calls these optimizer variables "slots".
		For momentum optimization, we need one momentum slot per model variable.
		"""
		for var in var_list:
			self.add_slot(var, "pw") #previous weights

	@tf.function(experimental_relax_shapes=True)
	def _resource_apply_dense(self, grad, var):
		"""Update the slots and perform one optimization step for one model variable
		"""
		var_dtype = var.dtype.base_dtype
		mom_weight = self._get_hyper('momentum_weight', var_dtype)
		lr_t = self._decayed_lr(var_dtype)
		pw_var = self.get_slot(var, "pw")

		new_var_m = var - grad * lr_t + mom_weight * (var - pw_var)
		pw_var.assign(var)
		var.assign(new_var_m)

	def _resource_apply_sparse(self, grad, var):
		raise NotImplementedError

	def get_config(self):
		base_config = super().get_config()
		return {
			**base_config,
			"learning_rate": self._serialize_hyperparameter("learning_rate"),
			'momentum_weight': self._serialize_hyperparameter('momentum_weight')
		}

	def _resource_apply_sparse(self, grad, var):
		raise NotImplementedError

class HeavyBall:
	def __init__(
		self, 
		step_size=0.0001, 
		momentum_weight=0.0002,
		model=None,
		loss=None,
		dataset=None
	):
		self.model = model
		self.loss = loss
		self.dataset = dataset
		self.step_size = step_size
		self.momentum_weight = momentum_weight
		self.d_optimizer = HeavyBallOpt(learning_rate=step_size, momentum_weight=momentum_weight)
		self.g_optimizer = HeavyBallOpt(learning_rate=step_size, momentum_weight=momentum_weight)
		self.name = 'HeavyBall (' + str(self.step_size) + ', ' + str(momentum_weight) + ')'

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