import tensorflow as tf


class _encoder(object):
    def __init__(self, ipt, enc_shp, activator):
        u'''
        Parameters
        ----------
        ipt : Tensor
            输入Tensor, shape为: [None, dims]
        enc_shp : list of int
            编/解码器每一层输出的dim
        activator : tf.nn 下的激活函数
            对于编码器, 默认使用 tf.nn.relu
            对于解码器, 默认使用 tf.nn.sigmoid
        '''
        self.input = ipt
        self.enc_lays = []
        self.enc_shp = enc_shp
        for shp in self.enc_shp:
            input = self.enc_lays[-1] if self.enc_lays else ipt
            tmp_lay = tf.keras.layers.Dense(shp, activation=activator)(input)
            self.enc_lays.append(tmp_lay)

        self.output = self.enc_lays[-1]


class Decoder(_encoder):
    def __init__(self, ipt, enc_shp):
        super(Decoder, self).__init__(ipt, enc_shp, tf.nn.sigmoid)


class Encoder(_encoder):
    def __init__(self, ipt, enc_shp):
        super(Encoder, self).__init__(ipt, enc_shp, tf.nn.sigmoid)