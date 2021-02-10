import numpy as np
import tensorflow as tf
import zipfile 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
import pickle

class FFHQ128:
	def __init__(self, size=128, zip_file_path='files/thumbnails128x128.zip', dtype='float16'):
		X_train = None
		if zip_file_path.endswith('.pickle'):
			f = open(zip_file_path,'rb')
			object_file = pickle.load(f)
			X_train = object_file['images']
			X_train = X_train * 2 - 1
		else:
			with zipfile.ZipFile(zip_file_path, 'r') as ziphandler:
				i = 0
				X_train = np.zeros((70000, size, size, 3), dtype=dtype)
				for filename in ziphandler.namelist():
					if not filename.endswith('.png'):
						continue
					img = ziphandler.read(filename)
					d = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
					d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
					if size != 128:
						d2 = cv2.resize(d, (size,size))
					X_train[i] = d2 / 127.5 - 1
					i += 1
					if i%500 == 0:
						print('Load dataset',i)

		self.X = X_train
		self.img_shape = (size, size, 3)
		self.channels = 3
		self.dtype = dtype
		self.name = 'FFHQ_'+str(size)+'x'+str(size)

	def batch(self, batch_size):
		idx = np.random.randint(0, self.X.shape[0], batch_size)
		imgs = self.X[idx]
		return tf.convert_to_tensor(imgs, dtype=tf.float32)