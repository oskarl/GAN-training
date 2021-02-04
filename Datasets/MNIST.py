import numpy as np
from keras.datasets import mnist
import tensorflow as tf

class MNIST:
	def __init__(self, dtype='float32'):
		(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
		X_train = X_train.astype(dtype) / 127.5 - 1.
		X_train = np.expand_dims(X_train, axis=3)

		self.X = X_train
		self.img_shape = (28, 28, 1)
		self.channels = 1
		self.dtype = dtype
		self.name = 'MNIST'

	def batch(self, batch_size):
		idx = np.random.randint(0, self.X.shape[0], batch_size)
		imgs = self.X[idx]
		return tf.convert_to_tensor(imgs, dtype=tf.float32)