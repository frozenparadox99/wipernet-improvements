import pytest
import tensorflow as tf
from keras.layers import Input
from keras.models import Model

from wipernet.components.discriminator import Discriminator1

# from discriminator_module import Discriminator1  # Uncomment and use if Discriminator1 is in a separate module

@pytest.fixture
def discriminator():
    return Discriminator1()

def test_discriminator_output_shape(discriminator):
    model = discriminator.discriminator
    input_tensor = tf.random.normal([1, 256, 256, 3])
    target_tensor = tf.random.normal([1, 256, 256, 3])
    output_tensor = model([input_tensor, target_tensor])
    assert output_tensor.shape == (1, 30, 30, 1), f"Expected output shape (1, 30, 30, 1), but got {output_tensor.shape}"

def test_conv2d(discriminator):
    conv_layer = discriminator.conv2d(filters=64, size=4, stride=2, name='test_conv')
    input_tensor = tf.random.normal([1, 256, 256, 6])
    output_tensor = conv_layer(input_tensor)
    assert output_tensor.shape == (1, 128, 128, 64), f"Expected output shape (1, 128, 128, 64), but got {output_tensor.shape}"

def test_downsample_block(discriminator):
    downsample_layer = discriminator.downsample_block(filters=64, size=4, name='test_downsample')
    input_tensor = tf.random.normal([1, 256, 256, 6])
    output_tensor = downsample_layer(input_tensor)
    assert output_tensor.shape == (1, 128, 128, 64), f"Expected output shape (1, 128, 128, 64), but got {output_tensor.shape}"

def test_batch_norm(discriminator):
    input_tensor = tf.random.normal([1, 256, 256, 64])
    output_tensor = discriminator.batch_norm(input_tensor)
    assert output_tensor.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"

def test_discriminator_loss(discriminator):
    real_output = tf.random.normal([1, 30, 30, 1])
    generated_output = tf.random.normal([1, 30, 30, 1])
    loss = discriminator.discriminator_loss(real_output, generated_output)
    assert isinstance(loss, tf.Tensor), "Loss should be a tensor"
    assert loss.shape == (), f"Loss should be a scalar, but got shape {loss.shape}"

if __name__ == "__main__":
    pytest.main()
