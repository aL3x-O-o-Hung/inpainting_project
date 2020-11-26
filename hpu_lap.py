from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.ops import math_ops

BASE_NUM_KERNELS = 64

print(tf.__version__)


class BatchNormRelu(tf.keras.layers.Layer):
    """Batch normalization + ReLu"""

    def __init__(self, name=None, dtype=None):
        super(BatchNormRelu, self).__init__(name=name)
        self.bnorm = tf.keras.layers.experimental.SyncBatchNormalization(momentum=0.9, dtype=dtype)

    def call(self, inputs, is_training):
        x = self.bnorm(inputs, training=is_training)
        x = tf.keras.activations.swish(x)
        return x


class Conv2DTranspose(tf.keras.layers.Layer):
    """Conv2DTranspose layer"""

    def __init__(self, output_channels, kernel_size, name=None, dtype=None):
        super(Conv2DTranspose, self).__init__(name=name)
        self.tconv1 = tf.keras.layers.Conv2DTranspose(
            filters=output_channels,
            kernel_size=kernel_size,
            strides=2,
            padding='same',
            activation=None,
            dtype=dtype
        )

    def call(self, inputs):
        x = self.tconv1(inputs)
        return x


class Conv2DFixedPadding(tf.keras.layers.Layer):
    """Conv2D Fixed Padding layer"""

    def __init__(self, filters, kernel_size, stride, name=None, dtype=None):
        super(Conv2DFixedPadding, self).__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=1,
            dilation_rate=1,
            padding=('same' if stride == 1 else 'valid'),
            activation=None,
            dtype=dtype
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        return x


class ConvBlock(tf.keras.layers.Layer):
    """Downsampling ConvBlock on Encoder side"""

    def __init__(self, filters, kernel_size=3, do_max_pool=True, name=None):
        super(ConvBlock, self).__init__(name=name)
        self.do_max_pool = do_max_pool
        self.conv1 = Conv2DFixedPadding(filters=filters,
                                        kernel_size=kernel_size,
                                        stride=1)
        self.brelu1 = BatchNormRelu()
        self.conv2 = Conv2DFixedPadding(filters=filters,
                                        kernel_size=kernel_size,
                                        stride=1)
        self.brelu2 = BatchNormRelu()
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=2,
                                                  strides=2,
                                                  padding='valid')

    def call(self, inputs, is_training):
        x = self.conv1(inputs)
        x = self.brelu1(x, is_training)
        x = self.conv2(x)
        x = self.brelu2(x, is_training)
        output_b = x
        if self.do_max_pool:
            x = self.max_pool(x)
        return x, output_b


class DownSampleBlock(tf.keras.layers.Layer):
    """Downsampling ConvBlock on Encoder side"""

    def __init__(self, filters, kernel_size=3, name=None):
        super(DownSampleBlock, self).__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=2,
            dilation_rate=1,
            padding='same',
            activation=None,
        )
        self.brelu1 = BatchNormRelu()
        self.brelu2 = BatchNormRelu()

    def call(self, inputs, is_training):
        output_b = self.brelu1(inputs, is_training)
        x = self.conv(inputs)
        x = self.brelu2(x, is_training)
        return x, output_b


class ResNetConvBlock(tf.keras.layers.Layer):
    """Downsampling ConvBlock on Encoder side"""

    def __init__(self, filters, kernel_size=3, name=None):
        super(ResNetConvBlock, self).__init__(name=name)
        self.filters = filters
        if self.filters <= 128:
            self.conv1 = Conv2DFixedPadding(filters=filters,
                                            kernel_size=kernel_size,
                                            stride=1)
            self.brelu1 = BatchNormRelu()
            self.conv2 = Conv2DFixedPadding(filters=filters,
                                            kernel_size=kernel_size,
                                            stride=1)
            self.brelu2 = BatchNormRelu()
        else:
            self.conv1 = Conv2DFixedPadding(filters=filters // 4,
                                            kernel_size=1,
                                            stride=1)
            self.brelu1 = BatchNormRelu()
            self.conv2 = Conv2DFixedPadding(filters=filters // 4,
                                            kernel_size=kernel_size,
                                            stride=1)
            self.brelu2 = BatchNormRelu()
            self.conv3 = Conv2DFixedPadding(filters=filters,
                                            kernel_size=1,
                                            stride=1)
            self.brelu3 = BatchNormRelu()

    def call(self, inputs, is_training):
        if self.filters <= 128:
            x = self.brelu1(inputs, is_training)
            x = self.conv1(x)
            x = self.brelu2(x, is_training)
            x = self.conv2(x)
        else:
            x = self.brelu1(inputs, is_training)
            x = self.conv1(x)
            x = self.brelu2(x, is_training)
            x = self.conv2(x)
            x = self.brelu3(x, is_training)
            x = self.conv3(x)
        x = inputs + x
        return x


class DeConvBlock(tf.keras.layers.Layer):
    """Upsampling DeConvBlock on Decoder side"""

    def __init__(self, filters, kernel_size=2, name=None):
        super(DeConvBlock, self).__init__(name=name)
        self.tconv1 = Conv2DTranspose(output_channels=filters,
                                      kernel_size=kernel_size)
        self.conv1 = Conv2DFixedPadding(filters=filters,
                                        kernel_size=3,
                                        stride=1)
        self.brelu1 = BatchNormRelu()
        self.conv2 = Conv2DFixedPadding(filters=filters,
                                        kernel_size=3,
                                        stride=1)
        self.brelu2 = BatchNormRelu()

    def call(self, inputs, output_b, is_training):
        x = self.tconv1(inputs)

        """Cropping is only used when convolution padding is 'valid'"""
        src_shape = output_b.shape[1]
        tgt_shape = x.shape[1]
        start_pixel = int((src_shape - tgt_shape) / 2)
        end_pixel = start_pixel + tgt_shape

        cropped_b = output_b[:, start_pixel:end_pixel, start_pixel:end_pixel, :]
        """Assumes that data format is NHWC"""
        x = tf.concat([cropped_b, x], axis=-1)

        x = self.conv1(x)
        x = self.brelu1(x, is_training)
        x = self.conv2(x)
        x = self.brelu2(x, is_training)
        return x


class ResNetDeConvBlock(tf.keras.layers.Layer):
    """Upsampling DeConvBlock on Decoder side"""

    def __init__(self, filters, kernel_size=2, name=None):
        super(ResNetDeConvBlock, self).__init__(name=name)
        self.tconv1 = Conv2DTranspose(output_channels=filters,
                                      kernel_size=kernel_size)
        self.conv1 = Conv2DFixedPadding(filters=filters,
                                        kernel_size=3,
                                        stride=1)
        self.brelu1 = BatchNormRelu()
        self.brelu2 = BatchNormRelu()

        self.brelu_list = []
        self.conv_list = []

        for i in range(3):
            self.brelu_list.append([])
            self.conv_list.append([])
            if filters <= 128:
                tmp_brelu = BatchNormRelu(name=name + '_brelu' + str(i + 1) + '_1')
                tmp_conv = Conv2DFixedPadding(filters=filters,
                                              kernel_size=3,
                                              stride=1,
                                              name=name + '_conv' + str(i + 1) + '_1')
                self.brelu_list[-1].append(tmp_brelu)
                self.conv_list[-1].append(tmp_conv)
                tmp_brelu = BatchNormRelu(name=name + '_brelu' + str(i + 1) + '_2')
                tmp_conv = Conv2DFixedPadding(filters=filters,
                                              kernel_size=3,
                                              stride=1,
                                              name=name + '_conv' + str(i + 1) + '_2')
                self.brelu_list[-1].append(tmp_brelu)
                self.conv_list[-1].append(tmp_conv)
            else:
                tmp_brelu = BatchNormRelu(name=name + '_brelu' + str(i + 1) + '_1')
                tmp_conv = Conv2DFixedPadding(filters=filters // 4,
                                              kernel_size=1,
                                              stride=1,
                                              name=name + '_conv' + str(i + 1) + '_1')
                self.brelu_list[-1].append(tmp_brelu)
                self.conv_list[-1].append(tmp_conv)
                tmp_brelu = BatchNormRelu(name=name + '_brelu' + str(i + 1) + '_3')
                tmp_conv = Conv2DFixedPadding(filters=filters // 4,
                                              kernel_size=3,
                                              stride=1,
                                              name=name + '_conv' + str(i + 1) + '_3')
                self.brelu_list[-1].append(tmp_brelu)
                self.conv_list[-1].append(tmp_conv)
                tmp_brelu = BatchNormRelu(name=name + '_brelu' + str(i + 1) + '_3')
                tmp_conv = Conv2DFixedPadding(filters=filters,
                                              kernel_size=1,
                                              stride=1,
                                              name=name + '_conv' + str(i + 1) + '_3')
                self.brelu_list[-1].append(tmp_brelu)
                self.conv_list[-1].append(tmp_conv)

    def call(self, inputs, output_b, is_training):
        x = self.tconv1(inputs)

        """Cropping is only used when convolution padding is 'valid'"""
        src_shape = output_b.shape[1]
        tgt_shape = x.shape[1]
        start_pixel = int((src_shape - tgt_shape) / 2)
        end_pixel = start_pixel + tgt_shape

        cropped_b = output_b[:, start_pixel:end_pixel, start_pixel:end_pixel, :]
        """Assumes that data format is NHWC"""
        x = tf.concat([cropped_b, x], axis=-1)
        x = self.conv1(x)
        x = self.brelu1(x, is_training)

        for i in range(len(self.brelu_list)):
            y = x
            for j in range(len(self.brelu_list[i])):
                y = self.brelu_list[i][j](y, is_training=is_training)
                y = self.conv_list[i][j](y)
            x = x + y

        x = self.brelu2(x, is_training)

        return x


class PriorBlock(tf.keras.layers.Layer):
    """calculating Prior Block"""

    def __init__(self, filters, name=None):  # filters: number of the layers incorporated into the decoder
        super(PriorBlock, self).__init__(name=name)
        self.conv = Conv2DFixedPadding(filters=filters * 2, kernel_size=1, stride=1)

    def call(self, inputs, is_training):
        x = 0.1 * self.conv(inputs)
        s = x.get_shape().as_list()[3]
        mean = x[:, :, :, :s // 2]
        # mean =tf.keras.activations.tanh(mean)
        logstd = x[:, :, :, s // 2:]
        logstd = 3.0 * tf.keras.activations.tanh(logstd)
        std = K.exp(logstd)
        # var = K.abs(logvar)
        return tf.concat([mean, std], axis=-1)


@tf.function
def prob_function(inputs):
    # For sample method
    ts = inputs.get_shape()
    s = ts.as_list()
    s[3] = int(s[3] / 2)
    dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
    if ts[0] is None:
        samp = dist.sample([1, s[1], s[2], s[3]])
    else:
        samp = dist.sample([ts[0], s[1], s[2], s[3]])
    dis = 0.5 * tf.math.multiply(samp, inputs[:, :, :, s[3]:])
    dis = tf.math.add(dis, inputs[:, :, :, 0:s[3]])
    return dis


class Prob(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(Prob, self).__init__(name=name)

    def call(self, inputs):
        ts = inputs.get_shape()
        s = ts.as_list()
        s[3] = int(s[3] / 2)
        dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        if ts[0] is None:
            samp = dist.sample([1, s[1], s[2], s[3]])
        else:
            samp = dist.sample([ts[0], s[1], s[2], s[3]])
        dis = tf.math.multiply(samp, inputs[:, :, :, s[3]:])
        dis = tf.math.add(dis, inputs[:, :, :, 0:s[3]])
        return dis


class Encoder(tf.keras.layers.Layer):
    """encoder of the network"""

    def __init__(self, num_layers, num_filters, name=None, use_resnet=False):
        super(Encoder, self).__init__(name=name)
        self.use_resnet = use_resnet
        if use_resnet:
            self.convs = []
            for i in range(num_layers):
                if i == 0:
                    self.convs.append([])
                    conv_temp = ConvBlock(filters=num_filters[i], do_max_pool=False, name=name + '_conv' + str(i + 1))
                    self.convs[-1].append(conv_temp)
                    conv_temp = tf.keras.layers.Conv2D(
                        num_filters[i + 1],
                        kernel_size=3,
                        strides=2,
                        dilation_rate=1,
                        padding='same',
                        activation=None,
                        name=name + '_down' + str(i + 1),
                    )
                    self.convs[-1].append(conv_temp)

                elif i < num_layers - 1:
                    self.convs.append([])
                    for j in range(3):
                        conv_temp = ResNetConvBlock(filters=num_filters[i],
                                                    name=name + '_conv' + str(i + 1) + '_' + str(j + 1))
                        self.convs[-1].append(conv_temp)
                    conv_temp = DownSampleBlock(filters=num_filters[i + 1], name=name + '_down' + str(i + 1))
                    self.convs[-1].append(conv_temp)

                else:
                    self.convs.append([])
                    for j in range(3):
                        conv_temp = ResNetConvBlock(filters=num_filters[i],
                                                    name=name + '_conv' + str(i + 1) + '_' + str(j + 1))
                        self.convs[-1].append(conv_temp)

        else:
            self.convs = []
            for i in range(num_layers):
                if i < num_layers - 1:
                    conv_temp = ConvBlock(filters=num_filters[i], name=name + '_conv' + str(i + 1))
                else:
                    conv_temp = ConvBlock(filters=num_filters[i], do_max_pool=False, name=name + '_conv' + str(i + 1))
                self.convs.append(conv_temp)

    def call(self, inputs, is_training):
        list_b = []
        x = inputs
        if self.use_resnet:
            for i in range(len(self.convs)):
                for j in range(len(self.convs[i]) - 1):
                    x = self.convs[i][j](x, is_training=is_training)
                if i == 0:
                    x, b = x
                    x = self.convs[i][-1](x)
                elif i < len(self.convs) - 1:
                    x, b = self.convs[i][-1](x, is_training=is_training)
                else:
                    b = x
                list_b.append(b)
        else:
            for i in range(len(self.convs)):
                x, b = self.convs[i](x, is_training=is_training)
                list_b.append(b)
        return x, list_b


class DecoderWithPriorBlockPosterior(tf.keras.layers.Layer):
    """decoder of the network with prior block in Posterior"""

    def __init__(self, num_layers, num_filters, num_filters_prior, name=None, use_resnet=False):
        super(DecoderWithPriorBlockPosterior, self).__init__(name=name)
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.num_filters_prior = num_filters_prior
        self.deconvs = []
        self.priors = []
        for i in range(num_layers):
            if use_resnet:
                self.deconvs.append(ResNetDeConvBlock(num_filters[i], name=name + '_dconv' + str(i)))
            else:
                self.deconvs.append(DeConvBlock(num_filters[i], name=name + '_dconv' + str(i)))
            self.priors.append(PriorBlock(num_filters_prior[i], name=name + 'prior' + str(i)))

    def call(self, inputs, blocks, is_training):
        x = inputs
        prior = []
        for i in range(self.num_layers):
            p = self.priors[i](x, is_training=is_training)
            prior.append(p)
            if i != self.num_layers - 1:
                x = tf.concat([x, p], axis=-1)
                x = self.deconvs[i](x, blocks[i], is_training=is_training)
        return prior


class DecoderWithPriorBlock(tf.keras.layers.Layer):
    """decoder of the network with prior block"""

    def __init__(self, num_layers, num_filters, num_filters_prior, name=None, use_resnet=False):
        super(DecoderWithPriorBlock, self).__init__(name=name)
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.num_filters_prior = num_filters_prior
        self.deconvs = []
        self.priors = []
        self.prob_function = Prob()
        for i in range(num_layers):
            if use_resnet:
                self.deconvs.append(ResNetDeConvBlock(num_filters[i], name=name + '_dconv' + str(i)))
            else:
                self.deconvs.append(DeConvBlock(num_filters[i], name=name + '_dconv' + str(i)))
            self.priors.append(PriorBlock(num_filters_prior[i], name=name + '_prior' + str(i)))

    def call(self, inputs, blocks, posterior_delta, is_training):
        x = inputs
        prior = []
        for i in range(self.num_layers):
            p = self.priors[i](x, is_training=is_training)
            prior.append(p)
            s = p.get_shape().as_list()[3]
            posterior = tf.concat([
                prior[i][:, :, :, :s // 2] + posterior_delta[i][:, :, :, :s // 2],
                prior[i][:, :, :, s // 2:] * posterior_delta[i][:, :, :, s // 2:],
            ], axis=-1)
            prob = self.prob_function(posterior)
            x = tf.concat([x, prob], axis=-1)
            x = self.deconvs[i](x, blocks[i], is_training=is_training)
        return x, prior

    def sample(self, x, blocks, is_training):
        for i in range(self.num_layers):
            p = self.priors[i](x, is_training=is_training)
            prob = prob_function(p)
            x = tf.concat([x, prob], axis=-1)
            x = self.deconvs[i](x, blocks[i], is_training=is_training)
        return x


class Decoder(tf.keras.layers.Layer):
    """decoder of the network"""

    def __init__(self, num_layers, num_prior_layers, num_filters, num_filters_in_prior, num_filters_prior,
                 name=None, use_resnet=False):
        """
        :param num_layers: number of layers in the non-prior part
        :param num_prior_layers: number of layers in the prior part
        :param num_filters: list of numbers of filters in different layers in non-prior part
        :param num_filters_in_prior: list of numbers of filters in different layers in non-prior part
        :param num_filters_prior: list of numbers of priors in different prior blocks
        :param name: name
        """
        super(Decoder, self).__init__(name=name)
        self.num_layers = num_layers
        self.num_filters_prior = num_filters_prior
        self.num_filters = num_filters
        self.num_filters_in_prior = num_filters_in_prior
        self.num_prior_layers = num_prior_layers
        self.prior_decode = DecoderWithPriorBlock(num_prior_layers,
                                                  num_filters_in_prior,
                                                  num_filters_prior,
                                                  name=name + '_with_prior',
                                                  use_resnet=use_resnet)
        self.tconvs = []
        self.generate = []
        for i in range(num_layers):
            if use_resnet:
                self.tconvs.append(ResNetDeConvBlock(num_filters[i], name=name + '_without_prior' + str(i)))
            else:
                self.tconvs.append(DeConvBlock(num_filters[i], name=name + '_without_prior' + str(i)))

    def call(self, inputs, b, posterior_delta, is_training):
        x = inputs
        x, prior = self.prior_decode(x, b[0:self.num_prior_layers], posterior_delta, is_training=is_training)
        for i in range(self.num_layers):
            x = self.tconvs[i](x, b[self.num_prior_layers + i], is_training=is_training)
        return x, prior

    def sample(self, x, b, is_training):
        x = self.prior_decode.sample(x, b[0:self.num_prior_layers], is_training=is_training)
        for i in range(self.num_layers):
            x = self.tconvs[i](x, b[self.num_prior_layers + i], is_training=is_training)
        return x


def residual_kl_gauss(y_delta, y_prior):
    # See NVAE paper
    s = y_delta.get_shape().as_list()[3]
    mean_delta = y_delta[:, :, :, 0:s // 2]
    std_delta = y_delta[:, :, :, s // 2:]
    # mean_prior = y_prior[:, :, :, 0:s // 2]
    std_prior = y_prior[:, :, :, s // 2:]
    first = math_ops.log(std_delta)
    second = 0.5 * math_ops.divide(K.square(mean_delta), K.square(std_prior))
    third = 0.5 * K.square(std_delta)
    loss = second + third - first - 0.5
    loss = tf.reduce_mean(loss)
    return loss


class HierarchicalProbUNet(tf.keras.Model):
    def __init__(self, num_layers, num_filters, num_prior_layers, num_filters_prior, rec, p, s, tv,
                 name=None, use_resnet=False):
        """
        :param num_layers: an integer, number of layers in the encoder side
        :param num_filters: a list, number of filters at each layer
        :param num_prior_layers: an integer, number of layers with prior blocks
        :param num_filters_prior: a list, number of filters at each prior blocks
        :param rec: a float, weight of reconstruction loss
        :param p: a list, weights of perceptual loss at each VGG layer
        :param s: a list, weights of style loss at each VGG layer
        :param tv: a float, weight of total variation loss
        :param name:
        """
        super(HierarchicalProbUNet, self).__init__(name=name)
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.num_prior_layers = num_prior_layers
        self.num_filters_decoder = num_filters[::-1][1:]
        self.num_filters_prior = num_filters_prior
        self.encoder = Encoder(num_layers,
                               num_filters,
                               name=name + '_encoder',
                               use_resnet=use_resnet)
        self.encoder_post = Encoder(num_layers,
                                    num_filters,
                                    name=name + '_encoder_post',
                                    use_resnet=use_resnet)
        self.decoder = Decoder(num_layers=num_layers - num_prior_layers - 1,
                               num_prior_layers=num_prior_layers,
                               num_filters=self.num_filters_decoder[num_prior_layers:],
                               num_filters_in_prior=self.num_filters_decoder[:num_prior_layers],
                               num_filters_prior=num_filters_prior,
                               name=name + '_decoder',
                               use_resnet=use_resnet)
        self.decoder_post = DecoderWithPriorBlockPosterior(num_prior_layers,
                                                           self.num_filters_decoder[:num_prior_layers],
                                                           num_filters_prior,
                                                           name=name + '_decoder_post',
                                                           use_resnet=use_resnet)
        self.conv = Conv2DFixedPadding(filters=3, kernel_size=1, stride=1, name=name + '_conv_final')
        self.VGG = VGG16()
        for layer in self.VGG.layers:
            layer.trainable = False
        self.VGGs = []
        for i in range(1, 6):
            if p[i - 1] > 0 or s[i - 1] > 0:
                name = 'block' + str(i) + '_conv1'
                vgg_input = self.VGG.input
                vgg_out = self.VGG.get_layer(name).output
                self.VGGs.append(tf.keras.Model(inputs=vgg_input, outputs=vgg_out))
                for layer in self.VGGs[i - 1].layers:
                    layer.trainable = False
            else:
                self.VGGs.append(None)
        self.rec = rec
        self.p = p
        self.s = s
        self.tv = tv
        self.upsamp = tf.keras.layers.UpSampling2D()
        self.downsamp = tf.keras.layers.AveragePooling2D()

    def get_gaussian_pyramid(self, mask, l):
        masks = [mask]
        for i in range(l - 1):
            mask = tfa.image.gaussian_filter2d(mask)
            mask = self.downsamp(mask)
            masks.append(mask)
        masks = masks[::-1]
        return masks

    # def get_laplacian_pyramid(self, p):
    #     py = []
    #     for i in range(len(p)):
    #         if i == 0:
    #             py.append(p[i])
    #         else:
    #             py.append(p[i] - tfa.image.gaussian_filter2d(self.upsamp(p[i - 1])))
    #     return py
    #
    # def build_from_pyramid(self, lp_input, lp_result, masks):
    #     final_res = []
    #     for i in range(len(lp_input)):
    #         if i == 0:
    #             final_res.append(lp_result[i] * masks[i] + lp_input[i] * (1 - masks[i]))
    #         else:
    #             final_res.append(lp_result[i] * masks[i] + lp_input[i] * (1 - masks[i]) +
    #                              tfa.image.gaussian_filter2d(self.upsamp(final_res[i - 1])))
    #     return final_res[len(lp_input) - 1]

    def reconstruction_loss(self, y_true, y_pred, weight):
        l = self.num_layers
        tg = self.get_gaussian_pyramid(y_true, l)
        pg = self.get_gaussian_pyramid(y_pred, l)
        # tl = self.get_laplacian_pyramid(tg)
        # pl = self.get_laplacian_pyramid(pg)
        for i in range(l - 1):
            if i == 0:
                loss = 0.5 * (4 ** i) * K.square(tg[i] - pg[i])
                loss = weight * tf.reduce_mean(loss)
                self.add_metric(loss, name='reconstruction loss' + str(i), aggregation='mean')
            else:
                temp_loss = 0.5 * (4 ** i) * K.square(tg[i] - pg[i])
                temp_loss = weight * tf.reduce_mean(temp_loss)
                self.add_metric(temp_loss, name='reconstruction loss' + str(i), aggregation='mean')
                loss += temp_loss
        temp_loss = 0.5 * (4 ** (l - 1)) * K.square(y_true - y_pred)
        temp_loss = weight * tf.reduce_mean(temp_loss)
        self.add_metric(temp_loss, name='reconstruction loss' + str(l - 1), aggregation='mean')
        loss += temp_loss
        return loss

    def perceptual_loss(self, y_true, y_pred, l, weight):
        loss = K.square(y_true - y_pred)
        loss = weight * tf.reduce_mean(loss)
        self.add_metric(loss, name='perceptual loss' + str(l), aggregation='mean')
        return loss

    def gram_matrix(self, x):
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]
        features = K.reshape(x, K.stack([B, C, H * W]))
        gram = K.batch_dot(features, features, axes=2)
        denominator = C * H * W
        gram = gram / K.cast(denominator, x.dtype)
        return gram

    def style_loss(self, y_true, y_pred, l, weight):
        y_true = self.gram_matrix(y_true)
        y_pred = self.gram_matrix(y_pred)
        loss = 0.5 * K.square(y_true - y_pred)
        loss = weight * tf.reduce_mean(loss)
        self.add_metric(loss, name='style loss' + str(l), aggregation='mean')
        return loss

    def deep_loss(self, y_true, y_pred, VGG_model, p, s):
        y_true = tf.image.resize(y_true, [224, 224])
        y_true = preprocess_input(y_true * 255)
        y_pred = tf.image.resize(y_pred, [224, 224])
        y_pred = preprocess_input(y_pred * 255)
        loss = 0
        for i in range(len(VGG_model)):
            if p[i] > 0 or s[i] > 0:
                y_true_ = VGG_model[i](y_true)
                y_pred_ = VGG_model[i](y_pred)
                loss += self.perceptual_loss(y_true_, y_pred_, i, p[i]) + self.style_loss(y_true_, y_pred_, i, s[i])
        return loss

    def total_variation_loss(self, y_pred):
        loss = tf.reduce_mean(tf.image.total_variation(y_pred))
        self.add_metric(loss, name='tv loss', aggregation='mean')
        return loss

    def total_loss(self, y_true, y_pred, VGG_model, rec, p, s, tv):
        loss = self.deep_loss(y_true, y_pred, VGG_model, p, s)
        if rec != 0:
            loss += self.reconstruction_loss(y_true, y_pred, rec)
        if tv > 0:
            loss += tv * self.total_variation_loss(y_pred)
        return loss

    def training_loss(self, y_true, y_pred, VGG_model, rec, p, s, tv):
        loss = self.total_loss(y_true, y_pred, VGG_model, rec, p, s, tv)
        return loss

    def call(self, inputs, is_training=True):
        x1 = inputs[:, :, :, 0:3]
        x2 = inputs[:, :, :, 4:7]
        original_input_x = x1
        ground_truth_x = x2
        mask = inputs[:, :, :, 3:4]
        x1 = tf.concat((x1, mask), axis=-1)
        x2 = tf.concat((x2, mask), axis=-1)
        x1, b_list1 = self.encoder(x1, is_training=is_training)
        b_list1 = b_list1[0:-1]
        x2, b_list2 = self.encoder_post(x2, is_training=is_training)
        b_list2 = b_list2[0:-1]
        b_list1.reverse()
        b_list2.reverse()
        posterior_delta = self.decoder_post(x2, b_list2[0:self.num_prior_layers], is_training=is_training)
        x1, prior = self.decoder(x1, b_list1, posterior_delta, is_training=is_training)
        x1 = self.conv(x1)
        x1 = tf.keras.activations.sigmoid(x1)
        x1 = x1 * mask + original_input_x * (1 - mask)
        # masks = self.get_gaussian_pyramid(mask, 4)
        # res_gaus = self.get_gaussian_pyramid(x1, 4)
        # inp_gaus = self.get_gaussian_pyramid(inputs[:, :, :, 0:3], 4)
        # res_ = self.get_laplacian_pyramid(res_gaus)
        # inp_ = self.get_laplacian_pyramid(inp_gaus)
        # x1 = self.build_from_pyramid(inp_, res_, masks)

        loss = self.training_loss(ground_truth_x, x1, self.VGGs, self.rec, self.p, self.s, self.tv)
        for i in range(len(prior)):
            if i == 0:
                los = residual_kl_gauss(posterior_delta[i], prior[i]) * (4 ** i)
                self.add_metric(los, name='kl_gauss' + str(i), aggregation='mean')
            else:
                temp = residual_kl_gauss(posterior_delta[i], prior[i]) * (4 ** i)
                self.add_metric(temp, name='kl_gauss' + str(i), aggregation='mean')
                los = math_ops.add(los, temp)
        self.add_loss(loss + los)
        return x1

    def sample(self, inputs, is_training):
        x, b_list = self.encoder(inputs, is_training=is_training)
        b_list = b_list[0:-1]
        b_list.reverse()
        x1 = self.decoder.sample(x, b_list, is_training=is_training)
        x1 = self.conv(x1)
        mask = inputs[:, :, :, 3:4]
        x1 = tf.keras.activations.sigmoid(x1)
        x1 = x1 * mask + inputs[:, :, :, 0:3] * (1 - mask)
        # masks = self.get_gaussian_pyramid(mask, 4)
        # res_gaus = self.get_gaussian_pyramid(x1, 4)
        # inp_gaus = self.get_gaussian_pyramid(inputs[:, :, :, 0:3], 4)
        # res_ = self.get_laplacian_pyramid(res_gaus)
        # inp_ = self.get_laplacian_pyramid(inp_gaus)
        # x1 = self.build_from_pyramid(inp_, res_, masks)
        return x1

    def reconstruct(self, inputs, is_training):
        x1 = inputs[:, :, :, 0:3]
        x2 = inputs[:, :, :, 4:7]
        original_input_x = x1
        ground_truth_x = x2
        mask = inputs[:, :, :, 3:4]
        x1 = tf.concat((x1, mask), axis=-1)
        x2 = tf.concat((x2, mask), axis=-1)
        x1, b_list1 = self.encoder(x1, is_training=is_training)
        b_list1 = b_list1[0:-1]
        x2, b_list2 = self.encoder_post(x2, is_training=is_training)
        b_list2 = b_list2[0:-1]
        b_list1.reverse()
        b_list2.reverse()
        posterior = self.decoder_post(x2, b_list2[0:self.num_prior_layers], is_training=is_training)
        x1, prior = self.decoder(x1, b_list1, posterior, is_training=is_training)
        x1 = self.conv(x1)
        x1 = tf.keras.activations.sigmoid(x1)
        x1 = x1 * mask + original_input_x * (1 - mask)
        return x1
