import numpy as np
from keras.datasets import cifar10
import tensorflow as tf

class CIFAR10:
	def __init__(self, dtype='float32'):
		(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
		X_train = X_train.astype(dtype) / 127.5 - 1.

		self.X = X_train
		self.img_shape = (32, 32, 3)
		self.channels = 3
		self.dtype = dtype
		self.name = 'CIFAR10'

	def batch(self, batch_size):
		idx = np.random.randint(0, self.X.shape[0], batch_size)
		imgs = self.X[idx]
		return tf.convert_to_tensor(imgs, dtype=tf.float32)