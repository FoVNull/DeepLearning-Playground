import tensorflow as tf
import tensorflow.keras.layers as L
from .DSN_folder.DSN_LOSS import DSN_LOSS
'''
DA的方法采用DSN模型，原文：
Bousmalis K, Trigeorgis G, Silberman N, Krishnan D, Erhan D. 
Domain separation networks. 
Advances in neural information processing systems. 2016;29:343-51.

L = L-task + αL-recon + βL-diff + γL-sim
L-task这里为知识蒸馏模型的loss

DSN参考了@WinChua的实现
https://github.com/WinChua/CDRTR/blob/master/CDRTR/core/DeepModel/DSN/model.py

'''



class A_Model:
    def __init__(self, output_dim, train_x, train_y, epochs, batch_size):
        self.tf_model = None
        self.output_dim = output_dim

        self.build_model()
        self.tf_model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)

        x = self.tf_model.get_layer(index=-2).output
        outputs = L.Softmax()(x / 3)

        self.a_model = tf.keras.Model(self.tf_model.input, outputs, name='A_Model')
        # self.a_model.summary()
        self.a_model.trainable = False

        self.a_model.compile(optimizer='adam',
                             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                             metrics=['accuracy'])

    def build_model(self):
        input_tensor = tf.keras.Input(shape=(224, 224, 3), name="img")

        tensor = input_tensor
        vgg19_layer = tf.keras.applications.vgg19.VGG19(include_top=True, 
                        weights='./models/vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                        )
        # 控制参数是否冻结
        for layer in vgg19_layer.layers:
           layer.trainable = False

        # (None, classes)
        tensor = vgg19_layer(tensor)

        layer_stack = [
            L.Dense(128, activation='relu'),
            L.Dense(128, activation='relu'),
            L.Dense(128, activation='relu'),
            L.Dense(2),
            L.Softmax()
        ]

        for layer in layer_stack:
            tensor = layer(tensor)

        self.tf_model = tf.keras.Model(inputs=input_tensor, outputs=tensor)
        # self.tf_model.summary()
        self.tf_model.compile(loss=tf.losses.sparse_categorical_crossentropy,
                              optimizer='adam',
                              metrics=['accuracy'])

    def return_model(self):
        return self.tf_model


class B_Model:
    def __init__(self):
        self.tf_model = None
        self.build_model()

    def build_model(self):
        input_tensor = tf.keras.Input(shape=(224, 224, 3), name="img")

        tensor = input_tensor

        vgg19_layer = tf.keras.applications.vgg19.VGG19(include_top=True, 
                weights='./models/vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                )
        # 控制参数是否冻结
        for layer in vgg19_layer.layers:
           layer.trainable = False

        tensor = vgg19_layer(tensor)

        layer_stack = [
            L.Dense(128, activation='relu'),
            L.Dense(128, activation='relu'),
            L.Dense(128, activation='relu'),
            L.Dense(2),
        ]

        for layer in layer_stack:
            tensor = layer(tensor)

        self.tf_model = tf.keras.Model(inputs=input_tensor, outputs=tensor, name='B_Model')
        # self.tf_model.summary()
        self.tf_model.compile(loss=tf.losses.sparse_categorical_crossentropy,
                              optimizer='adam',
                              metrics=['accuracy'])

    def return_model(self):
        return self.tf_model

# 我不确定是否需要训练encoder decoder，先写着，暂时不用
class Autocoder_Model:
    def __init__(self, coder_type):
        self.tf_model = None
        self.build_model()
        self.activ = 'relu' if coder_type == 'encode' else 'sigmoid' 

    def build_model(self):
        input_tensor = tf.keras.Input(shape=(224, 224, 3), name="img")

        tensor = tf.keras.layers.Dense(3, activation=self.activ)(input_tensor)

        self.tf_model = tf.keras.Model(inputs=input_tensor, outputs=tensor, name='encode_Model')
        self.tf_model.compile(loss=tf.losses.mse,
                              optimizer='adam')

    def return_model(self):
        return self.tf_model

class DA_Distilling(tf.keras.Model):
    def __init__(self, b_model, a_model, T, alpha):
        super(DA_Distilling, self).__init__()
        self.b_model = b_model
        self.a_model = a_model
        self.T = T
        self.alpha = alpha

    def train_step(self, data):
        input_x, y = data
        x = input_x['target_img']
        shared_x = x
        for k, v in input_x.items():  
            if k == 'target_img':
                continue
            shared_x = tf.concat([v, shared_x], axis=0)
        target_len, source_len = x.shape[0], shared_x.shape[0]-x.shape[0]
        domain_labels = tf.convert_to_tensor([0 for _ in range(target_len)] + [1 for _ in range(source_len)], dtype=tf.int16)

        softmax = L.Softmax()
        kld = tf.keras.losses.KLDivergence()

        with tf.GradientTape() as tape:
            logits = self.b_model(x)
            soft_labels = self.a_model(x)
            loss_value1 = self.compiled_loss(y, softmax(logits))
            loss_value2 = kld(soft_labels, softmax(logits / self.T))
            loss_value = self.alpha * loss_value2 + (1 - self.alpha) * loss_value1

            # dsn = DSN_LOSS(shared_x, domain_labels, [3], [3], [3], [3], [3])
            # loss_dsn = dsn.get_dsn_loss(1e-6, 1e-5, 1e-2)

            # loss_value += loss_dsn
            
        grads = tape.gradient(loss_value, self.b_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.b_model.trainable_weights))
        self.compiled_metrics.update_state(y, softmax(logits))
        return {'sum_loss': loss_value, 'loss1': loss_value1, 'loss2': loss_value2, }

    def test_step(self, data):
        input_x, y = data
        x = input_x['target_img']
        softmax = L.Softmax()
        logits = self.b_model(x)
        loss_value = self.compiled_loss(y, softmax(logits))

        return {'loss': loss_value}

    def call(self, inputs):
        print(inputs)
        exit(9)
        return self.b_model(inputs['target_img'])
