import os
import tensorflow as tf

def check_image_sizes(directory, expected_height, expected_width):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = tf.io.read_file(image_path)
            image = tf.image.decode_png(image)
            height, width = image.shape[:2]

            assert height == expected_height and width == expected_width, f"Image {filename} has incorrect size: {height}x{width} (expected {expected_height}x{expected_width})"

def test_image_sizes():
    output_dir = './artifacts/preprocessed'
    check_image_sizes(output_dir, 256, 256)
