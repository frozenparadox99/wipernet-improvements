train_rain_H_URL = 'artifacts/RainTrainH.zip'
train_rain_L_URL = 'artifacts/RainTrainL.zip'
test_rain_H_URL = 'artifacts/RainH.zip'
test_rain_L_URL = 'artifacts/RainL.zip'

unzip_dir_train_H = 'artifacts/data_ingestion'
unzip_dir_train_L = 'artifacts/data_ingestion'
unzip_dir_test_H = 'artifacts/data_ingestion_test'
unzip_dir_test_L = 'artifacts/data_ingestion_test'
train_dir = 'artifacts/train'
test_dir = 'artifacts/test'

output_dir = 'artifacts/output'

IMG_WIDTH: 256
IMG_HEIGHT: 256
BATCH_SIZE: 1

import os
import zipfile
import shutil
import re
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, DepthwiseConv2D, Add, Concatenate, GlobalAveragePooling2D, GlobalMaxPool2D, Dense, Reshape, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19, vgg19
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.layers import Input, concatenate, Conv2D, LeakyReLU, BatchNormalization, ZeroPadding2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy

import os
import datetime
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

def extract_zip_file(unzip_dir, config_file):
    """
    zip_file_path: str
    Extracts the zip file into the data directory
    Function returns None
    """
    unzip_path = unzip_dir
    os.makedirs(unzip_path, exist_ok=True)
    with zipfile.ZipFile(config_file, 'r') as zip_ref:
        zip_ref.extractall(unzip_path)

def combine_directories(images_dir, main_dir):
    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)
    ground_truth_counter = 1
    degraded_counter = 1


    for root, dirs, files in os.walk(main_dir):
        for sub_dir in dirs:
            image_pairs = {}
            sub_dir_path = os.path.join(root, sub_dir)
            for file_name in os.listdir(sub_dir_path):
                if file_name.endswith('.png'):
                    match = re.match(r'(norain|rain)-(\d+)\.png', file_name)
                    if match:
                        prefix, number = match.groups()
                        if number not in image_pairs:
                            image_pairs[number] = {}
                        image_pairs[number][prefix] = os.path.join(sub_dir_path, file_name)

            for counter, paths in sorted(image_pairs.items()):
                if 'norain' in paths:
                    dest_file_name = f"ground_truth_{ground_truth_counter}.png"
                    ground_truth_counter += 1
                    dest_file_path = os.path.join(images_dir, dest_file_name)
                    shutil.copy(paths['norain'], dest_file_path)

                if 'rain' in paths:
                    dest_file_name = f"degraded_{degraded_counter}.png"
                    degraded_counter += 1
                    dest_file_path = os.path.join(images_dir, dest_file_name)
                    shutil.copy(paths['rain'], dest_file_path)

def load(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image)
    image = tf.cast(image, tf.float32)
    return image

def filter_degraded_images(file_path):
    return tf.strings.regex_full_match(file_path, ".*degraded_.*\.png")

def load_image_train(image_file):
    # Load degraded image
    input_image = load(image_file)

    # Construct ground-truth image path
    real_image_path = tf.strings.regex_replace(image_file, 'degraded', 'ground_truth')

    # Load ground-truth image
    real_image = load(real_image_path)

    # Apply augmentations
    input_image, real_image = random_jitter(input_image, real_image)
    # input_image, real_image = resize(input_image, real_image, 256, 256)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file):
    # Load degraded image
    input_image = load(image_file)

    # Construct ground-truth image path
    real_image_path = tf.strings.regex_replace(image_file, 'degraded', 'ground_truth')

    # Load ground-truth image
    real_image = load(real_image_path)

    # Apply augmentations
    input_image, real_image = resize(input_image, real_image, 256, 256)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.BILINEAR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.BILINEAR)
    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, 256, 256, 3])
    return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image

@tf.function
def random_jitter(input_image, real_image):
    # randomly cropping to target size
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def preprocess(train_dataset, test_dataset):
    train_dataset = train_dataset.filter(filter_degraded_images)
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(1)
    train_dataset = train_dataset.batch(1)

    test_dataset = test_dataset.filter(filter_degraded_images)
    test_dataset = test_dataset.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(1)

    return train_dataset, test_dataset


vgg = VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

selected_layers = [
    'block3_conv4',
    'block4_conv4',
    'block5_conv3',
    'block5_conv4'
]
weightss = [1.0/32, 1.0/16, 1.0/8, 1.0/4]
outputs = [vgg.get_layer(name).output for name in selected_layers]
vgg_model = Model([vgg.input], outputs)

class Generator_1:
    def __init__(self, mode='train'):
        self.inputs = Input(shape=[None, None, 3])
        self.name = 'Generator_1/'
        print('=' * 50)
        print('input shape ::: ', self.inputs.shape)
        self.generator = self.build_generator()
        self.loss_object = BinaryCrossentropy(from_logits=True)

    def Channel_Attention(self, inputs, in_channels):
        AvgPool_xCat = GlobalAveragePooling2D()(inputs)
        MaxPool_xCat = GlobalMaxPool2D()(inputs)
        dense1 = Dense(in_channels // 8, activation='relu')(AvgPool_xCat)
        dense2 = Dense(in_channels, activation='relu')(dense1)
        dense3 = Dense(in_channels // 8, activation='relu')(MaxPool_xCat)
        dense4 = Dense(in_channels, activation='relu')(dense3)
        add = Add()([dense2, dense4])
        sig = Activation('sigmoid')(add)
        reshape = Reshape((1, 1, in_channels))(sig)
        return reshape

    def spatial_attention(self, input_feature):
        kernel_size = 7
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
        max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
        concat = tf.concat([avg_pool, max_pool], 3)
        concat = Conv2D(filters=1, kernel_size=[kernel_size, kernel_size], strides=[1, 1], padding="same", activation=None, kernel_initializer="he_normal", use_bias=False)(concat)
        concat = tf.sigmoid(concat)
        return concat * input_feature
        # return concat

    def SCA(self, inputs, in_channels):
        cam = self.Channel_Attention(inputs, in_channels)
        op = inputs * cam
        sam = self.spatial_attention(op)
        op = inputs * sam
        return op

    def generator_loss(self, disc_generated_output, gen_output, target, input_image):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        gen_loss_Edge = tf.reduce_mean(tf.abs(tf.image.sobel_edges(gen_output) - tf.image.sobel_edges(target)))
        SSIM_loss = 1 - tf.reduce_mean(tf.image.ssim(gen_output, target, max_val=1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03))
        vgg_gen_output = vgg_model(vgg19.preprocess_input(gen_output * 255.0))
        vgg_target = vgg_model(vgg19.preprocess_input(target * 255.0))
        vgg_input = vgg_model(vgg19.preprocess_input(input_image * 255.0))

        perceptual_loss_out = 0
        perceptual_loss_comp = 0
        contrastive_loss = 0
        for i in range(len(selected_layers)):
            perceptual_loss_out = tf.reduce_mean(tf.abs(vgg_gen_output[i] - vgg_target[i]))
            perceptual_loss_comp = tf.reduce_mean(tf.abs(vgg_gen_output[i] - vgg_input[i]))
            contrastive = perceptual_loss_out / (perceptual_loss_comp + 1e-7)
            contrastive_loss += weightss[i] * contrastive

        yuv_gen = tf.image.rgb_to_yuv(gen_output * 0.5 + 0.5)
        yuv_gt = tf.image.rgb_to_yuv(target * 0.5 + 0.5)
        yuv_loss = tf.reduce_mean(tf.abs(yuv_gen - yuv_gt))
        total_gen_loss = 0.01 * gan_loss + 1 * l1_loss + 5 * gen_loss_Edge + 20 * contrastive_loss + 10 * SSIM_loss + 5 * yuv_loss
        return total_gen_loss, gan_loss, l1_loss, contrastive_loss, gen_loss_Edge, SSIM_loss, yuv_loss

    def adaptive_mixup_feature_fusion_block(self, feature1, feature2, init_value=-0.80):
        weight = tf.Variable(init_value, dtype=tf.float32, trainable=True)
        sig = tf.math.sigmoid(weight)
        mix_factor = sig
        print("MIX FACTOR VALUE", mix_factor)
        out = feature1 * mix_factor + feature2 * (1 - mix_factor)
        return out

    def _residual_block(self, inputs, mixup_value=-0.80):
        x3 = Conv2D(16, 3, activation='relu', padding='same', use_bias=True)(inputs)
        x5 = Conv2D(16, 5, activation='relu', padding='same', use_bias=True)(inputs)
        concat1 = self.adaptive_mixup_feature_fusion_block(x3, x5, init_value=mixup_value)
        concat2 = self.adaptive_mixup_feature_fusion_block(x3, x5, init_value=mixup_value)
        x31 = Conv2D(16, 3, activation='relu', padding='same', use_bias=True)(concat1)
        x51 = Conv2D(16, 5, activation='relu', padding='same', use_bias=True)(concat2)
        concat = self.adaptive_mixup_feature_fusion_block(x31, x51, init_value=mixup_value)
        conv1 = Conv2D(inputs.shape[3], 3, activation='relu', padding='same', use_bias=True)(concat)
        add = Add()([conv1, inputs])
        return add

    def CARB(self, input_features):
        conv = Conv2D(16, 3, padding='same', activation='relu')(input_features)
        depthwise_conv = DepthwiseConv2D(3, padding='same', activation='relu')(conv)
        max_of_two = tf.math.maximum(conv, depthwise_conv)
        add = input_features + max_of_two
        return add

    def M_CARB(self, division_factor, input_features):
        downsampler = Conv2D(16, 3, padding='same', activation='relu', strides=division_factor)(input_features)
        CARB1 = self.CARB(downsampler)
        add1 = downsampler + CARB1
        CARB2 = self.CARB(add1)
        add2 = add1 + CARB2
        CARB3 = self.CARB(add2)
        return CARB3

    def M_CARB_NET(self, features):
        MCARB_ORIGINAL = self.M_CARB(1, features)
        MCARB_Half = self.M_CARB(2, features)
        MCARB_Quarter = self.M_CARB(4, features)
        Upsample_half = Conv2DTranspose(16, 3, padding='same', strides=2)(MCARB_Half)
        Upsample_Quarter = Conv2DTranspose(16, 3, padding='same', strides=4)(MCARB_Quarter)
        concatenate = Concatenate()([MCARB_ORIGINAL, Upsample_half, Upsample_Quarter])
        Dim_RED = Conv2D(16, 1, padding='same', activation='relu')(concatenate)
        return Dim_RED

    def Adaptive_Varying_Receptive_Fusion_Block(self, features):
        conv3 = Conv2D(16, 3, padding='same', activation='relu', dilation_rate=3)(features)
        conv5 = Conv2D(16, 3, padding='same', activation='relu', dilation_rate=5)(features)
        concatenate = self.adaptive_mixup_feature_fusion_block(conv3, conv5)
        conv_final = Conv2D(16, 1, padding='same', activation='relu')(concatenate)
        return conv_final

    def build_generator(self):
        def get_the_end_model(input_channel_num=3, feature_dim=16):
            def _rain_net(inputs):
                print('################## ORIGINAL DIMENSION STREAM #################')
                inputs1 = inputs
                x = Conv2D(16, (3, 3), padding="same", kernel_initializer="he_normal", use_bias=True, activation='relu')(inputs1)
                x0 = x
                MCARB1 = self.M_CARB_NET(x0)
                MCARB1_Conv = Conv2D(16, 1, padding='same', activation='relu')(MCARB1)
                MCARB2 = self.M_CARB_NET(MCARB1)
                MCARB2_Conv = Conv2D(16, 1, padding='same', activation='relu')(MCARB2)
                avrfb1 = self.Adaptive_Varying_Receptive_Fusion_Block(MCARB2)
                avrfb2 = self.Adaptive_Varying_Receptive_Fusion_Block(avrfb1)
                MCARB3 = self.M_CARB_NET(avrfb2)
                MCARB3_Conv = Conv2D(16, 1, padding='same', activation='relu')(MCARB3)
                MCARB4 = self.M_CARB_NET(MCARB3)
                MCARB4_Conv = Conv2D(16, 1, padding='same', activation='relu')(MCARB4)
                MCARB5 = self.M_CARB_NET(MCARB4)
                MCARB5_Conv = Conv2D(16, 1, padding='same', activation='relu')(MCARB5)

                x1 = self._residual_block(MCARB5)
                conv1 = Conv2D(16, 3, activation='relu', padding='same', use_bias=True)(x1)
                x1_att = self.SCA(conv1, 16)
                sub_x0_x1_att = x1_att - x0 - MCARB1_Conv

                x2 = self._residual_block(sub_x0_x1_att)
                conv2 = Conv2D(16, 3, activation='relu', padding='same', use_bias=True)(x2)
                x2_att = self.SCA(conv2, 16)
                x2_att = Conv2D(16, 3, padding='same', use_bias=True)(x2_att)
                sub_x1_att_x2_att = sub_x0_x1_att - x2_att - MCARB2_Conv

                x3 = self._residual_block(sub_x1_att_x2_att)
                conv3 = Conv2D(16, 3, activation='relu', padding='same', use_bias=True)(x3)
                x3_att = self.SCA(conv3, 16)
                sub_x2_att_x3_att = sub_x1_att_x2_att - x3_att - MCARB3_Conv

                x4 = self._residual_block(sub_x2_att_x3_att)
                conv4 = Conv2D(16, 3, activation='relu', padding='same', use_bias=True)(x4)
                x4_att = self.SCA(conv4, 16)
                x4_att = Conv2D(16, 3, padding='same', use_bias=True)(x4_att)
                sub_x4_att_x5_att = sub_x2_att_x3_att - x4_att - MCARB4_Conv

                x5 = self._residual_block(sub_x4_att_x5_att)
                conv5 = Conv2D(16, 3, activation='relu', padding='same', use_bias=True)(x5)
                x5_att = self.SCA(conv5, 16)
                sub_x5_att_x4_att = x5_att - sub_x4_att_x5_att - MCARB5_Conv

                x6 = self._residual_block(sub_x5_att_x4_att)
                finals = Conv2D(3, 1, padding="same", kernel_initializer="he_normal", activation='tanh')(x6)

                return finals

            inputs = Input(shape=(256, 256, input_channel_num), name='Rain_image')
            Rain = _rain_net(inputs)

            model = Model(inputs=inputs, outputs=Rain)
            return model

        model = get_the_end_model()
        return model

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


class GAN(object):
    """GAN class.
    Args:
        epochs: Number of epochs.
        path: path to folder containing images (training and testing)..
        mode: (train, test).
        output_path : output path for saving model
    """
    def __init__(self, epochs, path, mode, output_path):
        self.epochs = epochs
        self.path = path
        self.output_path = output_path
        os.path.join(self.output_path)
        self.lambda_value = 10
        self.gen1 = Generator_1()
        self.generator1 = self.gen1.generator
        self.print_info(self.generator1, 'Generator 1')

        self.disc1 = Discriminator1()
        self.discriminator1 = self.disc1.discriminator
        self.print_info(self.discriminator1, 'Discriminator 1')

        self.generator1_optimizer = Adam(5e-3, beta_1=0.5)
        self.discriminator1_optimizer = Adam(5e-3, beta_1=0.5)

        self.checkpoint_dir1 = os.path.join(self.output_path, 'training_checkpoints', 'gen1')
        self.checkpoint_prefix1 = os.path.join(self.checkpoint_dir1, "ckpt")
        self.checkpoint1 = tf.train.Checkpoint(generator1_optimizer=self.generator1_optimizer,
                                               discriminator1_optimizer=self.discriminator1_optimizer,
                                               generator1=self.generator1,
                                               discriminator1=self.discriminator1)

        log_dir = os.path.join(self.output_path, "logs")
        self.summary_writer = tf.summary.create_file_writer(log_dir + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    def generate_images(self, test_input, tar, number, mode='train'):
        derained = self.generator1(test_input, training=True)
        display_list = [test_input[0], derained[0], tar[0]]
        image = np.hstack([img for img in display_list])
        try:
            os.makedirs(os.path.join(self.output_path, mode), exist_ok=True)
        except:
            pass
        plt.imsave(os.path.join(self.output_path, mode, f'{number}_.png'), np.array((image * 0.5 + 0.5) * 255, dtype='uint8'))

    def print_info(self, object, name):
        print('=' * 50)
        text = f'Total Trainable parameters of {name} are :: {object.count_params()}'
        print(text)
        print('=' * 50)

    def train_step(self, input_image, target, epoch):
        with tf.GradientTape() as gen_tape1, tf.GradientTape() as disc_tape1:
            derained = self.generator1(input_image, training=True)
            disc1_real_output = self.discriminator1([input_image, target], training=True)
            disc1_generated_output = self.discriminator1([input_image, derained], training=True)
            gen1_total_loss, gen1_gan_loss, gen1_l1_loss, VGG_loss, gen_loss_Edge, SSIM_loss, yuv_loss = self.gen1.generator_loss(disc1_generated_output, derained, target, input_image)
            disc1_loss = self.disc1.discriminator_loss(disc1_real_output, disc1_generated_output)

        generator1_gradients = gen_tape1.gradient(gen1_total_loss, self.generator1.trainable_variables)
        discriminator1_gradients = disc_tape1.gradient(disc1_loss, self.discriminator1.trainable_variables)
        self.generator1_optimizer.apply_gradients(zip(generator1_gradients, self.generator1.trainable_variables))
        self.discriminator1_optimizer.apply_gradients(zip(discriminator1_gradients, self.discriminator1.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen1_total_loss', gen1_total_loss, step=epoch)
            tf.summary.scalar('gen1_gan_loss', gen1_gan_loss, step=epoch)
            tf.summary.scalar('gen1_l1_loss', gen1_l1_loss, step=epoch)
            tf.summary.scalar('disc1_loss', disc1_loss, step=epoch)

        outputs = {
            'gen1_total_loss': gen1_total_loss,
            'gen1_gan_loss': gen1_gan_loss,
            'gen1_l1_loss': gen1_l1_loss,
            'gen_loss_Edge': gen_loss_Edge,
            'VGG_loss': VGG_loss,
            'SSIM_loss': SSIM_loss,
            'yuv_loss': yuv_loss,
            'disc1_loss': disc1_loss,
        }

        return outputs

    def fit(self, train_ds, epochs, test_ds):
    # def fit(self, train_ds, epochs):
        for epoch in range(self.epochs):
            start = time.time()
            print(f"Checking for Epoch {epoch}")
            print(f"Epoch: {epoch + 1}")
            img_save = 0
            for n, (input_image, target) in tqdm(train_ds.enumerate()):
                outputs = self.train_step(input_image, target, epoch)
                if n % 10 == 0:
                    for k, (example_input, example_target) in tqdm(test_ds.take(5).enumerate()):
                        self.generate_images(example_input, example_target, img_save)
                        img_save += 1
                    print('=' * 50)
                    print(f'[!] gen1_total_loss :: {outputs["gen1_total_loss"]}')
                    print(f'[!] gen1_gan_loss :: {outputs["gen1_gan_loss"]}')
                    print(f'[!] gen1_l1_loss :: {outputs["gen1_l1_loss"]}')
                    print(f'[!] disc1_loss :: {outputs["disc1_loss"]}')
                    print(f'[!] VGG_loss :: {outputs["VGG_loss"]}')
                    print(f'[!] gen_loss_Edge :: {outputs["gen_loss_Edge"]}')
                    print(f'[!] SSIM LOSS :: {outputs["SSIM_loss"]}')
                    print(f'[!] YUV LOSS :: {outputs["yuv_loss"]}')
                    print('=' * 50)
                if n % 5000 == 0:
                    self.checkpoint1.save(file_prefix=self.checkpoint_prefix1)

            print(f'Time taken for epoch {epoch + 1} is {time.time() - start} sec')
            self.checkpoint1.save(file_prefix=self.checkpoint_prefix1)

    def test(self, dataset):
        self.checkpoint1.restore(tf.train.latest_checkpoint(self.checkpoint_dir1))
        print('Checkpoint restored !!!')
        for n, (example_input, example_target) in tqdm(dataset.enumerate()):
            self.generate_images(example_input, example_target, n, mode='test')
        print("Model Tested Successfully !!!!!")

    def load_checkpoint(self):
        self.checkpoint1.restore(tf.train.latest_checkpoint(self.checkpoint_dir1))

# extract_zip_file(unzip_dir_train_H, train_rain_H_URL)
# extract_zip_file(unzip_dir_train_L, train_rain_L_URL)
# extract_zip_file(unzip_dir_test_H, test_rain_H_URL)
# extract_zip_file(unzip_dir_test_L, test_rain_L_URL)

# combine_directories(train_dir, unzip_dir_train_H)
# combine_directories(test_dir, unzip_dir_test_H)

train_dataset = tf.data.Dataset.list_files(train_dir+ '/*.png')
test_dataset = tf.data.Dataset.list_files(test_dir+ '/*.png')

train_dataset, test_dataset = preprocess(train_dataset, test_dataset)

gan = GAN(epochs=3, path='./', mode='train', output_path=output_dir)

gan.fit(train_dataset, 3, test_dataset)