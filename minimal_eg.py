from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import Adam, SGD
import keras
import tensorflow as tf
import pickle
import json
import keras.backend as K
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import training_ops

import matplotlib.pyplot as plt

import sys
import math

import numpy as np

class GAN():
    def __init__(self, filename, description, latent_dim=100):
        self.filename = filename
        self.description = description

        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim

        optimizer = SGD(0.02)
        optimizer2 = SGD(0.02)
        self.optimizer = optimizer
        self.optimizer2 = optimizer2

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer2)

        self.loss_object = keras.losses.BinaryCrossentropy(from_logits=True)

    def loss(self, model, x, y, training):
        y_ = model(x, training=training)

        return self.loss_object(y_true=y, y_pred=y_)

    def grad(self, model, inputs, targets, only_trainable = False):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, training=True)
        alltrainablevars = list(filter(lambda x: not x.name.startswith('batch_normalization'), model.variables))
        return loss_value, tape.gradient(loss_value, (model.trainable_variables if only_trainable else alltrainablevars))

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        (X_train, _), (_, _) = mnist.load_data()

        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            print(epoch)
            # ---------------------
            #  Train Discriminator
            # ---------------------

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            noise = noise

            gen_imgs = self.generator.predict(noise)

            d_loss, grads = self.grad(self.discriminator, np.concatenate((imgs, gen_imgs), axis=0), np.concatenate((valid, fake), axis=0))

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss, grads_gen = self.grad(self.combined, noise, valid, only_trainable=True)

            alltrainablevars = list(filter(lambda x: not x.name.startswith('batch_normalization'), self.discriminator.variables))
            self.discriminator.optimizer.apply_gradients(zip(grads, alltrainablevars))
            self.combined.optimizer.apply_gradients(zip(grads_gen, self.combined.trainable_variables))

            # not same sample:
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            d_loss, grads_2 = self.grad(self.discriminator, np.concatenate((imgs, gen_imgs), axis=0), np.concatenate((valid, fake), axis=0))

            g_loss, grads_gen_2 = self.grad(self.combined, noise, valid, only_trainable=True)

            alltrainablevars = list(filter(lambda x: not x.name.startswith('batch_normalization'), self.discriminator.variables))
            self.discriminator.optimizer.apply_gradients(zip(-1 * grads, alltrainablevars))
            self.combined.optimizer.apply_gradients(zip(-1 * grads_gen, self.combined.trainable_variables))
            alltrainablevars = list(filter(lambda x: not x.name.startswith('batch_normalization'), self.discriminator.variables))
            self.discriminator.optimizer.apply_gradients(zip(grads_2, alltrainablevars))
            self.combined.optimizer.apply_gradients(zip(grads_gen_2, self.combined.trainable_variables))

            #print(K.eval(self.optimizer.iterations))

            if (epoch+1) % sample_interval == 0:
                # Plot the progress
                print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))

                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


gan = GAN('Dense_EG', 'MNIST', latent_dim=100)
gan.train(epochs=5000, batch_size=64, sample_interval=500)
