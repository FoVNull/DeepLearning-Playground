import logging
import pickle
import json
import numpy as np
from pandas.core.arrays import sparse
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.spatial.distance import pdist
from transformers import BertTokenizer

from utils import text2seq

logging.getLogger().setLevel(level=logging.DEBUG)
tokenizer = BertTokenizer('./model/hfl_chinese_bert_hf/vocab.txt')
token2id = json.load(open('./model/hfl_chinese_bert_hf/tokenizer.json', 'r', encoding='utf-8'))['model']['vocab']

def build_features(user_inputs):
    features_dic = pickle.load(open('./data/features_interface.pkl', 'rb'))
    input_features = {'sparse_data':[], 'dense_data':[], 'text':[]}
    for k, v in user_inputs['sparse'].items():
        feature = features_dic['sparse'][k][v] if v in features_dic['sparse'][k].keys() else max(features_dic['sparse'][k].values()) + 1
        input_features['sparse_data'].append(feature)

    for k, v in user_inputs['dense'].items():
        min_value, max_value = features_dic['dense'][k]
        scaled_value = MinMaxScaler().fit_transform(np.array([min_value, v, max_value]).reshape(-1, 1))[1][0]
        input_features['dense_data'].append(scaled_value)
    
    input_features['text'] = text2seq(user_inputs['text'])

    for k in input_features.keys():
        input_features[k] = np.array(input_features[k])

    return input_features

def build_new_features(path):
    dense_cols = ['旅游人数', '旅游时间', '旅游时长', '旅游花销']
    sparse_cols = ['旅游类别']
    data = pd.read_excel(path)

    features_interface_dic = {'sparse':{}, 'dense':{}}
    for sparse_col in sparse_cols:
        temp_onehot = LabelEncoder().fit_transform(data[sparse_col])
        features_interface_dic['sparse'][sparse_col] = {}
        for i in range(len(data[sparse_col])):
            features_interface_dic['sparse'][sparse_col][data[sparse_col][i]] = temp_onehot[i]
        data[sparse_col] = temp_onehot
    
    data[dense_cols] = data[dense_cols].fillna(0)
    for col in dense_cols:
        features_interface_dic['dense'][col] =  (min(data[col]), max(data[col]))
    data[dense_cols] = MinMaxScaler().fit_transform(data[dense_cols])

    new_features = {}
    for i in range(len(data)):
        features = []
        for col in dense_cols + sparse_cols:
            features.append(data[col][i])
        new_features[i] = features
    
    pickle.dump(new_features, open('./data/new_features.pkl', 'wb'))
    pickle.dump(features_interface_dic, open('./data/new_features_interface.pkl', 'wb'))

def calculate_sim(user_inputs):
    features = pickle.load(open('./data/new_features.pkl', 'rb'))
    features_dic = pickle.load(open('./data/new_features_interface.pkl', 'rb'))

    input_features = []

    for k, v in user_inputs['sparse'].items():
        feature = features_dic['sparse'][k][v] if v in features_dic['sparse'][k].keys() else max(features_dic['sparse'][k].values()) + 1
        input_features.append(feature)

    for k, v in user_inputs['dense'].items():
        min_value, max_value = features_dic['dense'][k]
        scaled_value = MinMaxScaler().fit_transform(np.array([min_value, v, max_value]).reshape(-1, 1))[1][0]
        input_features.append(scaled_value)
    
    # 余弦相似度cos，皮尔逊系数pearson
    def get_sim(x, y, method='cos'):
        X=np.vstack([x,y])
        simlarity = 0
        if method == 'cos':
            simlarity = 1-pdist(X,'cosine')
        if method == 'pearson':
            simlarity = np.corrcoef(X)[0][1]
        return simlarity[0]
    
    features_sim = {}
    for k, v in features.items():
        features_sim[k] = get_sim(input_features, v)
    ranking = sorted(features_sim.items(), key=lambda x:x[1], reverse=True)
    return ranking

if __name__ == '__main__':
    # 使用训练后的DeepFM模型获得预测结果
    # my_input = {'sparse':{'景点名':'白玉山景区', '级别':'4A', '开放时间':'全天开放', '用户名':'xxx'},
    #             'dense':{'评论数':2500, '总评分': 4.2},
    #             'text':'针不戳，住在山里真不戳'}
    # inputs = build_features(my_input)

    # model = tf.keras.models.load_model('./model/output')
    # pred = model.predict(inputs)
    # print(pred)

    # 使用路径表重整版
    # 构建特征对照表与数据库中的特征
    build_new_features('./data/路径表重整版.xlsx')
    my_input = {'sparse':{'旅游类别':'自驾游'},
                'dense':{'旅游人数':1, '旅游时间':1, '旅游时长':4, '旅游花销': 514}}
    ranking = calculate_sim(my_input)
    # 取前10
    for idx, sim in ranking[:10]:
        print(f'id: {idx}  similarity: {sim}')


