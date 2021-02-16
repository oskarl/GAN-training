import numpy as np
import tensorflow as tf
import zipfile 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
import pickle
import os

class ApplesOranges:
	def __init__(self, size=256, folder='files/apple2orange', dtype='float32'):
		apples = np.zeros((995, size, size, 3), dtype=dtype)
		oranges = np.zeros((1019, size, size, 3), dtype=dtype)
		i = 0
		for filename in os.listdir(folder+'/trainA'):
			img = cv2.imread(folder+'/trainA/'+filename)
			d = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			if size != 256:
				d = cv2.resize(d, (size,size))
			apples[i] = d / 127.5 - 1
			i += 1
		i = 0
		for filename in os.listdir(folder+'/trainB'):
			img = cv2.imread(folder+'/trainB/'+filename)
			d = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			if size != 256:
				d = cv2.resize(d, (size,size))
			oranges[i] = d / 127.5 - 1
			i += 1
		self.set1 = apples
		self.set2 = oranges
		self.img_shape = (size, size, 3)
		self.channels = 3
		self.dtype = dtype
		self.name = 'Apples & Oranges '+str(size)+'x'+str(size)

	def batch_set1(self, batch_size):
		idx = np.random.randint(0, self.set1.shape[0], batch_size)
		imgs = self.set1[idx]
		return tf.convert_to_tensor(imgs, dtype=tf.float32)

	def batch_set2(self, batch_size):
		idx = np.random.randint(0, self.set2.shape[0], batch_size)
		imgs = self.set2[idx]
		return tf.convert_to_tensor(imgs, dtype=tf.float32)