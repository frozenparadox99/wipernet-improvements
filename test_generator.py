import pytest
import tensorflow as tf

from wipernet.components.generator_1 import Generator_1

# from generator_module import Generator_1  # Uncomment and use if Generator_1 is in a separate module

@pytest.fixture
def generator():
    return Generator_1()

def test_generator_output_shape(generator):
    model = generator.generator
    input_tensor = tf.random.normal([1, 256, 256, 3])
    output_tensor = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"

def test_channel_attention(generator):
    input_tensor = tf.random.normal([1, 256, 256, 16])
    in_channels = 16
    channel_attention_output = generator.Channel_Attention(input_tensor, in_channels)
    assert channel_attention_output.shape == (1, 1, 1, in_channels), f"Expected output shape (1, 1, 1, {in_channels}), but got {channel_attention_output.shape}"

def test_spatial_attention(generator):
    input_tensor = tf.random.normal([1, 256, 256, 16])
    spatial_attention_output = generator.spatial_attention(input_tensor)
    assert spatial_attention_output.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {spatial_attention_output.shape}"

def test_SCA(generator):
    input_tensor = tf.random.normal([1, 256, 256, 16])
    in_channels = 16
    sca_output = generator.SCA(input_tensor, in_channels)
    assert sca_output.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {sca_output.shape}"

def test_residual_block(generator):
    input_tensor = tf.random.normal([1, 256, 256, 16])
    residual_block_output = generator._residual_block(input_tensor)
    assert residual_block_output.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {residual_block_output.shape}"

def test_adaptive_mixup_feature_fusion_block(generator):
    feature1 = tf.random.normal([1, 256, 256, 16])
    feature2 = tf.random.normal([1, 256, 256, 16])
    mixup_output = generator.adaptive_mixup_feature_fusion_block(feature1, feature2)
    assert mixup_output.shape == feature1.shape, f"Expected output shape {feature1.shape}, but got {mixup_output.shape}"

def test_CARB(generator):
    input_tensor = tf.random.normal([1, 256, 256, 16])
    carb_output = generator.CARB(input_tensor)
    assert carb_output.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {carb_output.shape}"

def test_M_CARB(generator):
    input_tensor = tf.random.normal([1, 256, 256, 16])
    mcarb_output = generator.M_CARB_NET(input_tensor)
    assert mcarb_output.shape == (1, 256, 256, 16), f"Expected output shape (1, 256, 256, 16), but got {mcarb_output.shape}"

def test_Adaptive_Varying_Receptive_Fusion_Block(generator):
    input_tensor = tf.random.normal([1, 256, 256, 16])
    avrfb_output = generator.Adaptive_Varying_Receptive_Fusion_Block(input_tensor)
    assert avrfb_output.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {avrfb_output.shape}"

def test_generator_loss(generator):
    disc_generated_output = tf.random.normal([1, 30, 30, 1])
    gen_output = tf.random.normal([1, 256, 256, 3])
    target = tf.random.normal([1, 256, 256, 3])
    input_image = tf.random.normal([1, 256, 256, 3])
    total_gen_loss, gan_loss, l1_loss, contrastive_loss, gen_loss_Edge, SSIM_loss, yuv_loss = generator.generator_loss(disc_generated_output, gen_output, target, input_image)
    assert isinstance(total_gen_loss, tf.Tensor), "total_gen_loss should be a tensor"
    assert isinstance(gan_loss, tf.Tensor), "gan_loss should be a tensor"
    assert isinstance(l1_loss, tf.Tensor), "l1_loss should be a tensor"
    assert isinstance(contrastive_loss, tf.Tensor), "contrastive_loss should be a tensor"
    assert isinstance(gen_loss_Edge, tf.Tensor), "gen_loss_Edge should be a tensor"
    assert isinstance(SSIM_loss, tf.Tensor), "SSIM_loss should be a tensor"
    assert isinstance(yuv_loss, tf.Tensor), "yuv_loss should be a tensor"

if __name__ == "__main__":
    pytest.main()
