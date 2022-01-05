from model.DeepFM import DeepFM
from utils import load_ctrip, preprocess

import tensorflow as tf
from tensorflow.keras import optimizers, losses
from sklearn.metrics import accuracy_score
import logging
import numpy as np

logging.getLogger().setLevel(level=logging.INFO)

if __name__ == '__main__':
    file_path = './data/携程景点评价.csv'

    gen_samples = False
    if gen_samples:
        preprocess(file_path)

    feature_columns, (X_train, y_train, text_train), (X_test, y_test, text_test) = load_ctrip(test_size=0.2)
    
    k = 10
    w_reg = 1e-4
    v_reg = 1e-4
    hidden_units = [256, 128, 64]
    output_dim = 1
    activation = 'relu'

    # sparse_num 代表了数值型特征的数量，注意 feature的格式应该为[离散型特征, 数值型特征]
    model = DeepFM(feature_columns[0], k, w_reg, v_reg, hidden_units, output_dim, activation, 
                    embedding_model_path='./model/hfl_chinese_bert_hf')
    
    optimizer = optimizers.SGD(0.01)
    loss = losses.BinaryCrossentropy()

    model.compile(loss=loss, optimizer=optimizer, metrics = ['accuracy', 'AUC', 'mae'])
    
    # 分割数值型与类别型变量
    sparse_x, dense_x = X_train[:, :feature_columns[1]], X_train[:, feature_columns[1]:]

    model.fit({"sparse_data": sparse_x, "dense_data": dense_x, "text": text_train}, y_train, epochs=10, batch_size=32)
    model.summary()

    sparse_x, dense_x = X_test[:, :feature_columns[1]], X_test[:, feature_columns[1]:]
    logloss, acc,  auc, mae = model.evaluate({"sparse_data": sparse_x, "dense_data": dense_x, "text": text_test}, y_test, batch_size=32)
    print(f'loss {logloss}\nACC: {acc}; AUC: {auc}; MAE{mae}')

    logging.info('saving model...')
    model.save('./model/output')

    # output = model.predict({"sparse_data": np.array([[10, 0, 8, 1821]]), "dense_data": np.array([[.7, 1.]]), "text": np.array([[1, 2, 3, 4]])})
    # print(output)
