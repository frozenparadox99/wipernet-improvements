import os
import datetime
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input, concatenate
from keras.models import Model
from keras.optimizers import Adam
from tqdm import tqdm

from wipernet.components.discriminator import Discriminator1
from wipernet.components.generator_1 import Generator_1
# from generator_module import Generator_1  # Uncomment and use if Generator_1 is in a separate module
# from discriminator_module import Discriminator1  # Uncomment and use if Discriminator1 is in a separate module
# You can uncomment the above imports and adjust the paths if these classes are in separate modules
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

        self.generator1_optimizer = Adam(2e-4, beta_1=0.5)
        self.discriminator1_optimizer = Adam(2e-4, beta_1=0.5)

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

    # def fit(self, train_ds, epochs, test_ds):
    def fit(self, train_ds, epochs):
        for epoch in range(self.epochs):
            start = time.time()
            print(f"Checking for Epoch {epoch}")
            print(f"Epoch: {epoch + 1}")
            img_save = 0
            for n, (input_image, target) in tqdm(train_ds.enumerate()):
                outputs = self.train_step(input_image, target, epoch)
                if n % 10 == 0:
                    # for k, (example_input, example_target) in tqdm(test_ds.take(5).enumerate()):
                    #     self.generate_images(example_input, example_target, img_save)
                    #     img_save += 1
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
