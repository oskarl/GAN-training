import numpy as np
import tensorflow as tf
import zipfile 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2

class CelebA:
	def __init__(self, size=32, zip_file_path='files/img_align_celeba.zip', dtype='float32'):
		with zipfile.ZipFile(zip_file_path, 'r') as ziphandler:
			i = 0
			X_train = np.zeros((200000, size, size, 3), dtype=dtype)
			for filename in ziphandler.namelist()[1:200000]:
				img = ziphandler.read(filename)
				d = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
				d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
				d2 = cv2.resize(d, (size,size))
				X_train[i] = d2 / 127.5 - 1
				i += 1
				if i%5000 == 0:
					print('Load dataset',i)

		self.X = X_train
		self.img_shape = (size, size, 3)
		self.channels = 3
		self.dtype = dtype
		self.name = 'CelebA_'+str(size)+'x'+str(size)

	def batch(self, batch_size):
		idx = np.random.randint(0, self.X.shape[0], batch_size)
		imgs = self.X[idx]
		return tf.convert_to_tensor(imgs, dtype=tf.float32)