import tensorflow as tf
from keras.layers import Input, Conv2D, Conv2DTranspose, DepthwiseConv2D, Add, Concatenate, GlobalAveragePooling2D, GlobalMaxPool2D, Dense, Reshape, Activation
from keras.models import Model

# Initialize VGG model for perceptual loss
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

selected_layers = [
    'block3_conv4',
    'block4_conv4',
    'block5_conv3',
    'block5_conv4'
]
weightss = [1.0/32, 1.0/16, 1.0/8, 1.0/4]
outputs = [vgg.get_layer(name).output for name in selected_layers]
vgg_model = tf.keras.Model([vgg.input], outputs)

class Generator_1:
    def __init__(self, mode='train'):
        self.inputs = Input(shape=[None, None, 3])
        self.name = 'Generator_1/'
        print('=' * 50)
        print('input shape ::: ', self.inputs.shape)
        self.generator = self.build_generator()
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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
        vgg_gen_output = vgg_model(tf.keras.applications.vgg19.preprocess_input(gen_output * 255.0))
        vgg_target = vgg_model(tf.keras.applications.vgg19.preprocess_input(target * 255.0))
        vgg_input = vgg_model(tf.keras.applications.vgg19.preprocess_input(input_image * 255.0))

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
        total_gen_loss = 0.01 * gan_loss + 10 * l1_loss + 10 * gen_loss_Edge + 20 * contrastive_loss + 15 * SSIM_loss + 15 * yuv_loss
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
