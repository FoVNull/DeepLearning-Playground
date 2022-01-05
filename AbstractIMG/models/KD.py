from sklearn import metrics as sklearn_metrics
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as L


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

        tensor = L.Conv2D(16, (4, 4), activation='relu', padding='valid')(tensor)

        layer_stack = [
            L.Flatten(),
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
        tensor = L.Conv2D(16, (4, 4), activation='relu', padding='valid')(tensor)

        query = L.GlobalMaxPooling2D()(tensor)
        att_seq = L.GlobalMaxPooling2D()(L.Attention()([tensor, tensor]))
        tensor = L.Concatenate(axis=-1)([query, att_seq])

        layer_stack = [
            # L.Flatten(),
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

        # self.tf_model.summary()

    def return_model(self):
        return self.tf_model


class Distilling(tf.keras.Model):
    def __init__(self, b_model, a_model, T, alpha):
        super(Distilling, self).__init__()
        self.b_model = b_model
        self.a_model = a_model
        self.T = T
        self.alpha = alpha

    def train_step(self, data):
        x, y = data
        softmax = L.Softmax()
        kld = tf.keras.losses.KLDivergence()
        with tf.GradientTape() as tape:
            logits = self.b_model(x)
            soft_labels = self.a_model(x)
            loss_value1 = self.compiled_loss(y, softmax(logits))
            loss_value2 = kld(soft_labels, softmax(logits / self.T))
            loss_value = self.alpha * loss_value2 + (1 - self.alpha) * loss_value1
        grads = tape.gradient(loss_value, self.b_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.b_model.trainable_weights))
        self.compiled_metrics.update_state(y, softmax(logits))
        return {'sum_loss': loss_value, 'loss1': loss_value1, 'loss2': loss_value2, }

    def test_step(self, data):
        x, y = data
        softmax = L.Softmax()
        logits = self.b_model(x)
        loss_value = self.compiled_loss(y, softmax(logits))

        return {'loss': loss_value}

    def call(self, inputs):
        return self.b_model(inputs)
