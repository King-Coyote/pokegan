import sys
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Conv2DTranspose, Deconvolution2D, Cropping2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from sklearn.externals import joblib
from functools import partial
from keras.layers.merge import _Merge
from keras.models import load_model
from skimage import io
import matplotlib.pyplot as plt
import math

BATCH_SIZE = 64
TRAINING_RATIO = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
CLIPPING_PARAM = 0.003

def wloss(actual, predicted):
    return K.mean(actual * predicted)

class PokeGAN:

    def build(self, generated_size=(48,48), alpha=0.2):
        # GENERATOR +++++++++++++++++++++++
        self.initial_size = (int(generated_size[0]/16), int(generated_size[1]/16))
        width, height = self.initial_size
        channels_count = 512
        initializer=RandomNormal(mean=0., stddev=CLIPPING_PARAM)
        
        self.generator = Sequential()
        self.generator.add(Dense(channels_count * width*height, input_shape=(100,), kernel_initializer=initializer))
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU(alpha))
        self.generator.add(Reshape([height, width, channels_count]))
        self.generator.add(Conv2DTranspose(channels_count//2, 5, strides=2,
            padding='same',
            data_format='channels_last',
            kernel_initializer=initializer))
        self.generator.add(BatchNormalization(momentum=0.5))
        self.generator.add(Activation('relu'))
        self.generator.add(Conv2DTranspose(channels_count//4, 5, strides=2,
            padding='same',
            data_format='channels_last',
            kernel_initializer=initializer))
        self.generator.add(BatchNormalization(momentum=0.5))
        self.generator.add(Activation('relu'))
        self.generator.add(Conv2DTranspose(channels_count//8, 5, strides=2,
            padding='same',
            data_format='channels_last',
            kernel_initializer=initializer))
        self.generator.add(BatchNormalization(momentum=0.5))
        self.generator.add(Activation('relu'))
        self.generator.add(Conv2DTranspose(3, 5, strides=2, padding='same', data_format='channels_last'))
        self.generator.add(Activation('tanh'))
        print("Generator Summary: ")
        self.generator.summary()
        #+++++++++++++++++++++++++++++++++++

        #DISCRIMINATOR +++++++++++++++++++++
        base = 64
        input_shape=(generated_size[0], generated_size[1], 3)
        self.discriminator = Sequential()
        self.discriminator.add(Convolution2D(
            base, 5, 
            strides=2, 
            kernel_initializer=initializer, 
            padding='same',
            data_format='channels_last',
            input_shape=input_shape))
        self.discriminator.add(LeakyReLU(alpha))
        self.discriminator.add(Convolution2D(
            base * 2, 5, 
            strides=2,
            kernel_initializer=initializer,
            padding='same', 
            data_format='channels_last'))
        self.discriminator.add(LeakyReLU(alpha))
        # self.discriminator.add(Dropout(0.25))
        self.discriminator.add(Convolution2D(
            base * 4, 5, 
            strides=2, 
            kernel_initializer=initializer, 
            padding='same',
            data_format='channels_last'))
        self.discriminator.add(LeakyReLU(alpha))
        # self.discriminator.add(Dropout(0.25))
        self.discriminator.add(Convolution2D(
            base * 8, 5, 
            strides=2, 
            kernel_initializer=initializer, 
            padding='same',
            data_format='channels_last'))
        self.discriminator.add(LeakyReLU(alpha))
        # self.discriminator.add(Dropout(0.25))
        self.discriminator.add(Flatten())
        self.discriminator.add(Dense(units=1, activation=None))
        self.discriminator.summary()
        #+++++++++++++++++++++++++++++++++++
        # d_optim = Adam(0.0001, beta_1=0.5, beta_2=0.9)
        # g_optim = Adam(0.0001, beta_1=0.5, beta_2=0.9)
        d_optim = RMSprop(lr=0.00005)
        g_optim = RMSprop(lr=0.00005)

        self.discriminator.compile(loss=wloss, optimizer = d_optim, metrics=None)
        self.generator.compile(loss=wloss, optimizer=g_optim, metrics=None)
        # z = Input(shape=(100,))
        # image = self.generator(z)
        # self.discriminator.trainable = False
        # valid = self.discriminator(image)

        # self.gan = Model(z, valid)
        # self.gan.compile(loss='binary_crossentropy', optimizer=g_optim)
        self.gan = Sequential()
        self.discriminator.trainable = False
        self.gan.add(self.generator)
        self.gan.add(self.discriminator)
        self.gan.compile(loss=wloss, optimizer=g_optim, metrics=None)

    def train(self, dump_filename, epochs=100):
        training_data = joblib.load(dump_filename)

        g_loss = []
        d_loss = []
        
        #zero_y = np.zeros((BATCH_SIZE, 1), dtype=np.float64)

        for epoch in range(epochs):
            np.random.shuffle(training_data)
            print("Epoch is", epoch)
            print("Number of batches", int(training_data.shape[0]/BATCH_SIZE))
            index = 0
            while index < int(training_data.shape[0]/BATCH_SIZE):
                for disc_index in range(TRAINING_RATIO):
                    noise = np.random.normal(-1, 1, size=(BATCH_SIZE,100))
                    if (index+1)*BATCH_SIZE >= training_data.shape[0]:
                        break
                    real_images = training_data[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
                    # if not (np.max(real_images) <= 1.0 and np.min(real_images) >= -1.0):
                    #     real_images = (real_images / 255.0).astype(np.float64)
                    #     real_images = real_images - 1.0
                    generated_images = self.generator.predict(noise)
                    real_y = np.ones((BATCH_SIZE,1), dtype=np.float64)
                    fake_y = np.ones((BATCH_SIZE,1), dtype=np.float64) * -1.0

                    self.discriminator.trainable = True

                    for layer in self.discriminator.layers:
                        weights = layer.get_weights()
                        weights = [np.clip(w, -1.0*CLIPPING_PARAM, CLIPPING_PARAM) for w in weights]
                        layer.set_weights(weights)

                    d_loss_real = self.discriminator.train_on_batch(real_images, real_y)
                    d_loss_fake = self.discriminator.train_on_batch(generated_images, fake_y)

                    d_loss.append(d_loss_real - d_loss_fake)
                    index += 1

                self.discriminator.trainable = False
                noise = np.random.normal(-1, 1, size=(BATCH_SIZE,100))
                fake_y = np.ones((BATCH_SIZE,1),dtype=np.float64)
                g_loss.append(self.gan.train_on_batch(noise, fake_y))
                print("epoch: {}, dloss: {}, gloss: {}".format(epoch, d_loss[-1], g_loss[-1]))
            sample = self.generator.predict(np.random.normal(-1, 1, (10, 100)))
            genned = sample[0]
            for i in range(1,10):
                genned = np.concatenate((genned, sample[i]), axis=1)
            io.imsave('generated/epoch_{}.png'.format(epoch), genned)
        plt.figure(1)
        plt.subplot(211)
        plt.plot(
            range(len(g_loss)), g_loss, 'b',
            range(len(d_loss)), d_loss, 'r',
        )
        plt.show()

    def predict(self, num, save=False, filename='generated/samples.png'):

        if save:
            samples = self.generator.predict(np.random.normal(-1, 1, (num, 100)))
            genned = samples[0]
            for i in range(1,num):
                genned = np.concatenate((genned, samples[i]), axis=1)
            io.imsave(filename, genned)
        else:
            while True:
                cmd = input("Enter q to quit. Otherwise, press enter to generate.")
                if cmd == 'q': 
                    return
                samples = self.generator.predict(np.random.normal(-1, 1, (num, 100)))
                genned = samples[0]
                for i in range(1,num):
                    genned = np.concatenate((genned, samples[i]), axis=1)
                io.imshow(genned)
                plt.show()

    def save(self, filename):
        self.discriminator.save(filename + '_disc.h5')
        print('saved discriminator as ' + filename + '_disc.h5')
        self.generator.save(filename + '_gen.h5')
        print('saved generator as ' + filename + '_gen.h5')
        # self.gan.save(filename + '_gan.h5')
        # print('saved gan as ' + filename + '_gan.h5')

    def load(self, filename):
        self.discriminator = keras.models.load_model(filename + '_disc.h5',custom_objects={'wloss': wloss})
        print('loaded discriminator!')
        self.generator = keras.models.load_model(filename + '_gen.h5', custom_objects={'wloss':wloss})
        print('loaded generator!')
        self.gan = Sequential()
        self.discriminator.trainable = False
        self.gan.add(self.generator)
        self.gan.add(self.discriminator)
        self.gan.compile(loss=wloss, optimizer=RMSprop(lr=0.00005), metrics=None)
        print('compiled GAN! Ready for generation.')
        # self.gan = keras.models.load_model(filename + '_gan.h5')
        # print('loaded gan!')
