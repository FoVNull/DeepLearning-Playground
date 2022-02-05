import tensorflow as tf
import tensorflow.keras.losses as losses
from .AutoCoder import Encoder, Decoder

def logitRegression(ipt):
    # W = tf.Variable(tf.zeros([ipt.shape[-1], 1]), dtype=tf.float32)
    # bias = tf.Variable(0.0, dtype=tf.float32)
    # y = tf.nn.sigmoid(tf.matmul(ipt, W) + bias)
    # return y
    ipt = tf.keras.layers.GlobalMaxPool2D()(ipt) 
    output = tf.keras.layers.Dense(1, activation='sigmoid')(ipt)
    return output


class DSN_LOSS:
    def __init__(self, ipt, domain_label,
                 source_enc_shp, target_enc_shp,
                 source_dec_shp, target_dec_shp,
                 share_enc_shp, source_rating=None
                 ):
        u'''
        构造DSN模型, 这里的实现没有考虑source domain对rating的损失, 只实现了
        以下几个loss:
            * 重构损失: 领域共有编码器与私有编码器之和经过解码器重构后的损失 # recon
            * 相似损失: 领域共有编码器与私有编码器之间的相似程度, 利用两个编码器 # diff
                        输出矩阵乘积的F范数进行衡量
            * 混淆损失: 共有编码器在不同领域的输出经过一个领域分类器,通过梯度反转
                        最大化分类误差 # sim
        DSN在source domain的编码输出采用:
            self.srcShrOut
        该向量包含为共享编码器对source domain用户向量的编码输出
        Parameters
        ----------
        ipt : Tensor
        domain_label : Tensor
        source_enc_shp, target_enc_shp, source_dec_shp, target_dec_shp : list of int
        source_rating : Tensor
        '''
        self.source_enc_shp = source_enc_shp
        self.source_dec_shp = source_dec_shp
        self.target_enc_shp = target_enc_shp
        self.target_dec_shp = target_dec_shp
        self.share_enc_shp = share_enc_shp
        self.source_rating = source_rating

        self.domain_label = domain_label
        self.ipt = ipt

    def get_dsn_loss(self, alpha, beta, gamma):

        # 依据领域标签构造两个domain_maks用于过滤ipt
        self.source_mask = tf.equal(self.domain_label, 1)
        # self.src_ipt = tf.boolean_mask(self.ipt, self.source_mask)
        # self.srcPriEnc = Encoder(self.src_ipt, self.source_enc_shp)

        self.target_mask = tf.equal(self.domain_label, 0)
        # self.tgt_ipt = tf.boolean_mask(self.ipt, self.target_mask)
        # self.tgtPriEnc = Encoder(self.tgt_ipt, self.target_enc_shp)

        self.sharedEnc = Encoder(self.ipt, self.share_enc_shp)
        self.srcShrOut = tf.boolean_mask(self.sharedEnc.output, self.source_mask)
        self.tgtShrOut = tf.boolean_mask(self.sharedEnc.output, self.target_mask)

        # self.srcHidden = self.srcPriEnc.output + self.srcShrOut
        # self.tgtHidden = self.tgtPriEnc.output + self.tgtShrOut

        # self.srcDec = Decoder(self.srcHidden, self.source_dec_shp)

        # self.tgtDec = Decoder(self.tgtHidden, self.target_dec_shp)

        # self.srcRstLoss = losses.mean_squared_error(self.src_ipt, self.srcDec.output)
        # self.tgtRstLoss = losses.mean_squared_error(self.tgt_ipt, self.tgtDec.output)
        # self.RstLoss = tf.reduce_sum(self.srcRstLoss) + tf.reduce_sum(self.tgtRstLoss)

        # 共享编码器输出不同领域相似损失
        self.domainProb = logitRegression(self.sharedEnc.output)
        # tf.stop_gradient使得参数的梯度不会在反向过程中传播,实现GRL梯度反转
        # 反向传播过程中只有-self.domainProb向后传播, 传递为- d{domainPorb}/d{x}
        # 向前传播的值为 -domainProb + domainProb + domainProb = domainProb
        self.domainProb = -self.domainProb + tf.stop_gradient(self.domainProb + self.domainProb)
        self.domainLoss = losses.binary_crossentropy(
            tf.expand_dims(self.domain_label, -1), self.domainProb)
        self.domainLoss = tf.reduce_sum(self.domainLoss)

        # 领域私有部分与共有部分差异损失: L_loss
        # self.srcDiffLoss = tf.norm(
        #         tf.matmul(self.srcPriEnc.output, self.srcShrOut, transpose_b=True)) 
        # self.tgtDiffLoss = tf.norm(
        #         tf.matmul(self.tgtPriEnc.output, self.tgtShrOut, transpose_b=True))
        # self.DiffLoss = self.srcDiffLoss + self.tgtDiffLoss
        # self.loss = alpha*self.RstLoss + gamma*self.domainLoss + beta*self.DiffLoss
        self.loss = gamma*self.domainLoss

        return self.loss