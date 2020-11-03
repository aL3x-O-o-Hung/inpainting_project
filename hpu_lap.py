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
        self.bnorm = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                        renorm=True,
                                                        renorm_momentum=0.9,
                                                        dtype=dtype)

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
            activation=tf.keras.activations.swish,
            dtype=dtype
        )

    def call(self, inputs):
        x = self.tconv1(inputs)
        return x


class Conv2DFixedPadding(tf.keras.layers.Layer):
    """Conv2D Fixed Padding layer"""

    def __init__(self, filters, kernel_size, stride, name=None, dtype=None):
        super(Conv2DFixedPadding, self).__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            kernel_size,
                                            strides=1,
                                            dilation_rate=1,
                                            padding=('same' if stride == 1 else 'valid'),
                                            activation=None,
                                            dtype=dtype
                                            # kernel_initializer=tf.keras.initializers.ones
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


class PriorBlock(tf.keras.layers.Layer):
    """calculating Prior Block"""

    def __init__(self, filters, name=None):  # filters: number of the layers incorporated into the decoder
        super(PriorBlock, self).__init__(name=name)
        self.conv = Conv2DFixedPadding(filters=filters * 2, kernel_size=1, stride=1)

    def call(self, inputs):
        x = self.conv(inputs)
        s = x.get_shape().as_list()[3]
        mean = x[:, :, :, 0:s // 2]
        mean = math.e * tf.keras.activations.tanh(mean)
        logvar = x[:, :, :, s // 2:]
        std = 0.1 + math.e * tf.keras.activations.sigmoid(logvar)
        # var = K.exp(logvar)
        # var = K.abs(logvar)
        return tf.concat([mean, std], axis=-1)


@tf.function
def prob_function(inputs):
    s = inputs.get_shape().as_list()
    s[3] = int(s[3] / 2)
    dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
    samp = dist.sample([1, s[1], s[2], s[3]])
    dis = tf.math.multiply(samp, inputs[:, :, :, s[3]:])
    dis = tf.math.add(dis, inputs[:, :, :, 0:s[3]])
    return dis


class Prob(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(Prob, self).__init__(name=name)

    def call(self, inputs):
        s = inputs.get_shape().as_list()
        s[3] = int(s[3] / 2)
        dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        samp = dist.sample([1, s[1], s[2], s[3]])
        dis = tf.math.multiply(samp, inputs[:, :, :, s[3]:])
        dis = tf.math.add(dis, inputs[:, :, :, 0:s[3]])
        return dis


class Encoder(tf.keras.layers.Layer):
    """encoder of the network"""

    def __init__(self, num_layers, num_filters, name=None):
        super(Encoder, self).__init__(name=name)
        self.convs = []
        for i in range(num_layers):
            if i < num_layers - 1:
                conv_temp = ConvBlock(filters=num_filters[i], name=name + '_conv' + str(i + 1))
            else:
                conv_temp = ConvBlock(filters=num_filters[i], do_max_pool=False, name=name + '_conv' + str(i + 1))
            self.convs.append(conv_temp)

    def call(self, inputs, is_training=True):
        list_b = []
        x = inputs
        for i in range(len(self.convs)):
            x, b = self.convs[i](x, is_training=is_training)
            list_b.append(b)
        return x, list_b


class DecoderWithPriorBlockPosterior(tf.keras.layers.Layer):
    """decoder of the network with prior block in Posterior"""

    def __init__(self, num_layers, num_filters, num_filters_prior, name=None):
        super(DecoderWithPriorBlockPosterior, self).__init__(name=name)
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.num_filters_prior = num_filters_prior
        self.deconvs = []
        self.priors = []
        self.prob_function = Prob()
        for i in range(num_layers):
            self.deconvs.append(DeConvBlock(num_filters[i], name=name + '_dconv' + str(i)))
            self.priors.append(PriorBlock(num_filters_prior[i], name=name + 'prior' + str(i)))

    def call(self, inputs, blocks, is_training=True):
        x = inputs
        prior = []
        prob = []
        for i in range(self.num_layers):
            p = self.priors[i](x)
            prior.append(p)
            p = self.prob_function(p)
            prob.append(p)
            if i != self.num_layers - 1:
                x = tf.concat([x, p], axis=-1)
                x = self.deconvs[i](x, blocks[i], is_training=is_training)
        return prior, prob


class DecoderWithPriorBlock(tf.keras.layers.Layer):
    """decoder of the network with prior block"""

    def __init__(self, num_layers, num_filters, num_filters_prior, name=None):
        super(DecoderWithPriorBlock, self).__init__(name=name)
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.num_filters_prior = num_filters_prior
        self.deconvs = []
        self.priors = []
        self.prob_function = Prob()
        for i in range(num_layers):
            self.deconvs.append(DeConvBlock(num_filters[i], name=name + '_dconv' + str(i)))
            self.priors.append(PriorBlock(num_filters_prior[i], name=name + '_prior' + str(i)))

    def call(self, inputs, blocks, prob, is_training=True):
        x = inputs
        prior = []
        for i in range(self.num_layers):
            p = self.priors[i](x)
            prior.append(p)
            x = tf.concat([x, prob[i]], axis=-1)
            x = self.deconvs[i](x, blocks[i], is_training=is_training)
        return x, prior

    def sample(self, x, blocks, is_training=False):
        for i in range(self.num_layers):
            p = self.priors[i](x)
            prob = prob_function(p)
            x = tf.concat([x, prob], axis=-1)
            x = self.deconvs[i](x, blocks[i], is_training=is_training)
        return x


class Decoder(tf.keras.layers.Layer):
    """decoder of the network"""

    def __init__(self, num_layers, num_prior_layers, num_filters, num_filters_in_prior, num_filters_prior, name=None):
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
                                                  name=name + '_with_prior')
        self.tconvs = []
        self.generate = []
        for i in range(num_layers):
            self.tconvs.append(DeConvBlock(num_filters[i], name=name + '_without_prior'))

    def call(self, inputs, b, prob, is_training=True):
        x = inputs
        x, prior = self.prior_decode(x, b[0:self.num_prior_layers], prob, is_training=is_training)
        for i in range(self.num_layers):
            x = self.tconvs[i](x, b[self.num_prior_layers + i], is_training=is_training)
        return x, prior

    def sample(self, x, b, is_training=False):
        x = self.prior_decode.sample(x, b[0:self.num_prior_layers], is_training=is_training)
        for i in range(self.num_layers):
            x = self.tconvs[i](x, b[self.num_prior_layers + i], is_training=is_training)
        return x


def kl_gauss(y_true, y_pred):
    s = y_true.get_shape().as_list()[3]
    mean_true = y_true[:, :, :, 0:s // 2]
    var_true = y_true[:, :, :, s // 2:]
    mean_pred = y_pred[:, :, :, 0:s // 2]
    var_pred = y_pred[:, :, :, s // 2:]
    first = math_ops.log(math_ops.divide(var_pred, var_true))
    second = math_ops.divide(var_true + K.square(mean_true - mean_pred), var_pred)
    loss = first + second - 1
    loss = K.flatten(loss * 0.5)
    loss = tf.reduce_mean(loss)
    return loss


class HierarchicalProbUNet(tf.keras.Model):
    def __init__(self, num_layers, num_filters, num_prior_layers, num_filters_prior, rec, p, s, tv, name=None):
        '''
        :param num_layers: an integer, number of layers in the encoder side
        :param num_filters: a list, number of filters at each layer
        :param num_prior_layers: an integer, number of layers with prior blocks
        :param num_filters_prior: a list, number of filters at each prior blocks
        :param rec: a float, weight of reconstruction loss
        :param p: a list, weights of perceptual loss at each VGG layer
        :param s: a list, weights of style loss at each VGG layer
        :param tv: a float, weight of total variation loss
        :param name:
        '''
        super(HierarchicalProbUNet, self).__init__(name=name)
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.num_prior_layers = num_prior_layers
        self.num_filters_decoder = num_filters[::-1]
        self.num_filters_prior = num_filters_prior
        self.encoder = Encoder(num_layers,
                               num_filters,
                               name=name + '_encoder')
        self.encoder_post = Encoder(num_layers,
                                    num_filters,
                                    name=name + '_encoder_post')
        self.decoder = Decoder(num_layers=num_layers - num_prior_layers - 1,
                               num_prior_layers=num_prior_layers,
                               num_filters=self.num_filters_decoder[num_prior_layers + 1:],
                               num_filters_in_prior=self.num_filters_decoder[1:num_prior_layers + 1],
                               num_filters_prior=num_filters_prior,
                               name=name + '_decoder')
        self.decoder_post = DecoderWithPriorBlockPosterior(num_prior_layers,
                                                           self.num_filters_decoder[1:num_prior_layers + 1],
                                                           num_filters_prior,
                                                           name=name + '_decoder_post')
        self.conv = Conv2DFixedPadding(filters=3, kernel_size=1, stride=1, name=name + '_conv_final')
        self.VGG = VGG16()
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

    def get_laplacian_pyramid(self, p):
        py = []
        for i in range(len(p)):
            if i == 0:
                py.append(p[i])
            else:
                py.append(p[i] - tfa.image.gaussian_filter2d(self.upsamp(p[i - 1])))
        return py

    def build_from_pyramid(self, lp_input, lp_result, masks):
        final_res = []
        for i in range(len(lp_input)):
            if i == 0:
                final_res.append(lp_result[i] * masks[i] + lp_input[i] * (1 - masks[i]))
            else:
                final_res.append(lp_result[i] * masks[i] + lp_input[i] * (1 - masks[i]) +
                                 tfa.image.gaussian_filter2d(self.upsamp(final_res[i - 1])))
        return final_res[len(lp_input) - 1]

    def reconstruction_loss(self, y_true, y_pred, weight):
        l = 5
        tg = self.get_gaussian_pyramid(y_true, l)
        pg = self.get_gaussian_pyramid(y_pred, l)
        tl = self.get_laplacian_pyramid(tg)
        pl = self.get_laplacian_pyramid(pg)
        for i in range(l):
            if i == 0:
                loss = 2 ** i * K.square(tl[i] - pl[i])
                loss = tf.reduce_mean(loss)
            else:
                temp_loss = 2 ** i * K.square(tl[i] - pl[i])
                loss += tf.reduce_mean(temp_loss)
        loss = loss / (2 ** l - 1)
        temp_loss = 0.5 * K.square(y_true - y_pred)
        loss += tf.reduce_mean(temp_loss)
        loss = weight * loss
        self.add_metric(loss, name='reconstruction loss', aggregation='mean')
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
        prior2, prob = self.decoder_post(x2, b_list2[0:self.num_prior_layers], is_training=is_training)
        x1, prior1 = self.decoder(x1, b_list1, prob, is_training=is_training)
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
        for i in range(len(prior1)):
            if i == 0:
                los = kl_gauss(prior2[i], prior1[i])
                self.add_metric(los, name='kl_gauss' + str(i), aggregation='mean')
            else:
                temp = kl_gauss(prior2[i], prior1[i])
                self.add_metric(temp, name='kl_gauss' + str(i), aggregation='mean')
                los = math_ops.add(los, temp)
        self.add_loss(loss + los)
        return x1

    def sample(self, inputs, is_training=False):
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
