import tensorflow as tf
from keras.layers import Input, concatenate, Conv2D, LeakyReLU, BatchNormalization, ZeroPadding2D
from keras.models import Model, Sequential
from keras.losses import BinaryCrossentropy

class Discriminator1:

    def __init__(self, mode='train'):
        if mode == 'train':
            self.inp = Input(shape=[256, 256, 3], name='input_image')
            self.tar = Input(shape=[256, 256, 3], name='target_image')
        else:
            self.inp = Input(shape=[None, None, 3], name='input_image')
            self.tar = Input(shape=[None, None, 3], name='target_image')
        self.input_ = concatenate([self.inp, self.tar], axis=3)
        self.name = 'Discriminator1/'
        self.discriminator = self.build_discriminator()

    def conv2d(self, filters, size, stride=2, name='conv2d'):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = Sequential()
        result.add(Conv2D(filters, size, strides=stride,
                          padding='same',
                          kernel_initializer=initializer,
                          use_bias=False,
                          name=self.name + name))
        return result

    def downsample_block(self, filters=16, size=3, apply_batchnorm=True, name='downsample'):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = Sequential()
        result.add(Conv2D(filters, size, strides=2, padding='same',
                          kernel_initializer=initializer, use_bias=False,
                          name='Conv_' + name))
        result.add(LeakyReLU())
        return result

    def batch_norm(self, tensor):
        return BatchNormalization(axis=3, epsilon=1e-5, momentum=0.1,
                                  gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(tensor)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        loss_object = BinaryCrossentropy(from_logits=True)
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    def build_discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)
        print('################## Build Discriminator 1 ##################')

        down1 = self.downsample_block(filters=32, size=4, name='DownSample1_')(self.input_)
        print('down1 ::: {}'.format(down1.shape))

        down2 = self.downsample_block(filters=64, size=4, name='DownSample2_')(down1)
        print('down2 ::: {}'.format(down2.shape))

        down3 = self.downsample_block(filters=128, size=4, name='DownSample3_')(down2)
        print('down3 ::: {}'.format(down3.shape))

        zero_pad1 = ZeroPadding2D()(down3)
        conv = Conv2D(512, 4, strides=1,
                      kernel_initializer=initializer,
                      use_bias=False)(zero_pad1)

        batchnorm1 = BatchNormalization()(conv)

        leaky_relu = LeakyReLU()(batchnorm1)

        zero_pad2 = ZeroPadding2D()(leaky_relu)

        last = Conv2D(1, 4, strides=1,
                      kernel_initializer=initializer)(zero_pad2)

        return Model(inputs=[self.inp, self.tar], outputs=last)
