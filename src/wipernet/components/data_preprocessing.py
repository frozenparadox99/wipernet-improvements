import os
import tensorflow as tf
from wipernet.entity.config_entity import DataPreProcessingConfig
from wipernet import logger
from pathlib import Path

class DataPreProcessing:
    def __init__(self, config: DataPreProcessingConfig):
        self.train_dataset = tf.data.Dataset.list_files(config.train_dir+ '/*.png')
        self.test_dataset = tf.data.Dataset.list_files(config.test_dir+ '/*.png')
        self.config = config

    def load(self, image_path):
        logger.info(image_path)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image)
        image = tf.cast(image, tf.float32)
        return image

    def filter_degraded_images(self, file_path):
        return tf.strings.regex_full_match(file_path, ".*degraded_.*\.png")

    def load_image_train(self, image_file):
        logger.info(image_file)
        # Load degraded image
        input_image = self.load(image_file)

        # Construct ground-truth image path
        real_image_path = tf.strings.regex_replace(image_file, 'degraded', 'ground_truth')

        # Load ground-truth image
        real_image = self.load(real_image_path)

        # Apply augmentations
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image
    
    def load_image_test(self, image_file):
        logger.info(image_file)
        # Load degraded image
        input_image = self.load(image_file)

        # Construct ground-truth image path
        real_image_path = tf.strings.regex_replace(image_file, 'degraded', 'ground_truth')

        # Load ground-truth image
        real_image = self.load(real_image_path)

        # Apply augmentations
        input_image, real_image = self.resize(input_image, real_image, self.config.params_image_height, self.config.params_image_width)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def preprocess(self):
        self.train_dataset = self.train_dataset.filter(self.filter_degraded_images)
        self.train_dataset = self.train_dataset.map(self.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
        self.train_dataset = self.train_dataset.shuffle(1)
        self.train_dataset = self.train_dataset.batch(self.config.params_batch_size)

        self.test_dataset = self.test_dataset.filter(self.filter_degraded_images)
        self.test_dataset = self.test_dataset.map(self.load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
        self.test_dataset = self.test_dataset.batch(self.config.params_batch_size)

    def resize(self, input_image, real_image, height, width):
        input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.BILINEAR)
        real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.BILINEAR)
        return input_image, real_image

    def random_crop(self, input_image, real_image):
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2, self.config.params_image_height, self.config.params_image_width, 3])
        return cropped_image[0], cropped_image[1]

    def normalize(self, input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1
        return input_image, real_image

    @tf.function
    def random_jitter(self, input_image, real_image):
        # randomly cropping to target size
        input_image, real_image = self.random_crop(input_image, real_image)

        if tf.random.uniform(()) > 0.5:
            # random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def save_preprocessed_images(self):
        output_dir = self.config.output_dir
        test_output_dir = 'artifacts/preprocessed-test'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)

        for input_image, real_image in self.train_dataset.take(1):
            for i in range(len(input_image)):
                input_image_path = os.path.join(output_dir, f'degraded_{i}.png')
                real_image_path = os.path.join(output_dir, f'ground_truth_{i}.png')

                tf.io.write_file(input_image_path, tf.image.encode_png(tf.cast((input_image[i] + 1) * 127.5, tf.uint8)))
                tf.io.write_file(real_image_path, tf.image.encode_png(tf.cast((real_image[i] + 1) * 127.5, tf.uint8)))
        
        for input_image, real_image in self.test_dataset.take(1):
            for i in range(len(input_image)):
                input_image_path = os.path.join(test_output_dir, f'degraded_{i}.png')
                real_image_path = os.path.join(test_output_dir, f'ground_truth_{i}.png')

                tf.io.write_file(input_image_path, tf.image.encode_png(tf.cast((input_image[i] + 1) * 127.5, tf.uint8)))
                tf.io.write_file(real_image_path, tf.image.encode_png(tf.cast((real_image[i] + 1) * 127.5, tf.uint8)))


