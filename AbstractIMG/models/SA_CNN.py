from sklearn import metrics as sklearn_metrics
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as L


class SA_CNN:
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.tf_model = None

    def build_model(self):
        img = tf.keras.Input(shape=(224, 224, 3), name="img")
        resnet_layer = tf.keras.applications.resnet50.ResNet50(classes=1000)
        res_tensor = resnet_layer(img)

        img_tensor = img
        img_stack = [
            L.Conv2D(32, (4, 4), activation='relu', padding='valid'),
            # L.MaxPooling2D(),
            # L.Dropout(rate=0.1),
        ]
        for img_layer in img_stack:
            img_tensor = img_layer(img_tensor)
        
        query = L.GlobalMaxPooling2D()(img_tensor)
        att_seq = L.GlobalMaxPooling2D()(L.Attention()([img_tensor, img_tensor]))
        input_tensor = L.Concatenate(axis=-1)([query, att_seq])

        output_tensor = L.Dense(self.output_dim, activation='sigmoid', name="output0")(input_tensor)
        self.tf_model = tf.keras.Model(inputs=img, outputs=output_tensor)
        self.tf_model.summary()

        self.tf_model.compile(loss=tf.losses.sparse_categorical_crossentropy,
                              optimizer='adam',
                              metrics=['accuracy'])

    def fit(self, train_x, train_y, epochs, batch_size):
        self.tf_model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)
    
    def evaluate(self, test_x, test_y, batch_size, digits=4):
        y_pred = self.tf_model.predict(test_x, batch_size=batch_size)
        y_pred = [np.argmax(onehot) for onehot in y_pred]
        report = sklearn_metrics.classification_report(test_y,
                                                        y_pred,
                                                        output_dict=True,
                                                        digits=digits)
        for k, v in report.items():
            print(f'{k} {v}')
        cm = sklearn_metrics.confusion_matrix(test_y, y_pred)
        print(cm)
        return report