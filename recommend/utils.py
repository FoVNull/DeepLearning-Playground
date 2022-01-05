import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import re
import numpy as np
from tqdm import tqdm
import pickle
import logging
import json

from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


logging.getLogger().setLevel(level=logging.DEBUG)
tokenizer = BertTokenizer('./model/hfl_chinese_bert_hf/vocab.txt')
token2id = json.load(open('./model/hfl_chinese_bert_hf/tokenizer.json', 'r', encoding='utf-8'))['model']['vocab']

user_feature = []
spot_feature = []

def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'feat': feat}

def text2seq(texts):
    token_seqs = []
    for text in texts:      
        token_seqs.append([token2id[token] for token in tokenizer.tokenize(text)])
    # maxlen = 512
    token_seqs = pad_sequences(token_seqs, padding="post", truncating="post", value=0, maxlen=64, dtype='int32')

    return token_seqs

def preprocess(file_path, embed_dim=8):
    data = pd.read_csv(file_path)

    # 将评论文本转成序列
    data['评论正文'] = data['评论正文'].fillna('评论')
    comments_seqs = text2seq(data['评论正文'])

    output = []
    # 特征对照表，便于后续自定义输入源数据得到数值型特征
    features_interface_dic = {'sparse':{}, 'dense':{}}

    # 处理离散型变量
    sparse_features = ['景点名', '级别', '开放时间', '用户名']
    data[sparse_features] = data[sparse_features].fillna('-1')
    sparse_size = [len(set(data[sparse_col])) for sparse_col in sparse_features]
    for sparse_col in sparse_features:
        temp_onehot = LabelEncoder().fit_transform(data[sparse_col])
        features_interface_dic['sparse'][sparse_col] = {}
        for i in range(len(data[sparse_col])):
            features_interface_dic['sparse'][sparse_col][data[sparse_col][i]] = temp_onehot[i]
        data[sparse_col] = temp_onehot

    # 数值化后的景点列表
    spot_list = set(data['景点名'])
    
    logging.debug(f'处理数值变量...')
    dense_features = ['评论数', '总评分']
    data[dense_features] = data[dense_features].fillna(0)

    for i in tqdm(range(len(data))):
        for col_name in dense_features + ['评分']:
            try:
                data[col_name][i] = float(''.join(re.findall(r'[0-9.]', data[col_name][i])))
            except:
                data[col_name][i] = float(data[col_name][i])
                
    for dense_col in dense_features:
        # 记录离散型变量的范围
        features_interface_dic['dense'][dense_col] =  (min(data[dense_col]), max(data[dense_col]))

    # 归一化
    data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])

    # DeepFM所需要的feature结构信息
    feature_columns = ([[denseFeature(feat) for feat in dense_features]] + \
           [[sparseFeature(feat, data[feat].nunique(), embed_dim) for feat in sparse_features]], len(sparse_features))
    
    # 构建用户-景点矩阵用于构建补充负例，同时记录已知的正负样本
    user2spot, spot2feature = {}, {}
    logging.debug(f'构建样本...')
    stat = [0, 0]
    for i in tqdm(range(len(data))):
        user, spot = data['用户名'][i], data['景点名'][i]
        if user in user2spot:
            user2spot[data['用户名'][i]].add(spot)
        else:
            user2spot[data['用户名'][i]] = {spot}
        
        # 景点通用特征
        spot2feature[spot] = [data[col_name][i] for col_name in ['景点名', '级别', '开放时间', '评论数', '总评分']]
        
        # 根据评分确定样本正负
        label = 1 if data['评分'][i] > 3. else 0
        stat[label] += 1
        sample = (np.array([data[col_name][i] for col_name in sparse_features + dense_features]), comments_seqs[i], label)
        output.append(sample)

    '''
    # 负例不足时构建补充负例，对于每个用户构建n个负例，n=景点总数-去过的景点
    for u, s in user2spot.items():
        for spot in spot_list:
            if s == spot:
                continue
            
            # '景点名', '级别', '开放时间' + 用户名 + '评论数', '总评分' + 评分, 标签
            negative_sample = (spot2feature[spot][:3] + [u] + spot2feature[spot][3:] + [0], 0)
            output.append(negative_sample)
            stat[0] += 1
    '''
    logging.debug(f'### neg: {stat[0]}; pos: {stat[1]}\n ###')

    pickle.dump(feature_columns, open('./model/feature_columns.pkl', 'wb'))
    pickle.dump(output, open('./data/ctrip_ctr_data.pkl', 'wb'))

    pickle.dump(features_interface_dic, open('./data/features_interface.pkl', 'wb'))


def load_ctrip(test_size=0.2):
    feature_columns = pickle.load(open('./model/feature_columns.pkl', 'rb'))
    samples = pickle.load(open('./data/ctrip_ctr_data.pkl', 'rb'))

    # 样本乱序
    np.random.seed(0)
    np.random.shuffle(samples)

    x, y, texts =  [], [], [] 
    for feature, text, label in samples:
        texts.append(text)
        x.append(feature)
        y.append(label)
    x = np.array(x)
    y = np.array(y)
    texts = np.array(texts)

    x_train, x_test, y_train, y_test, text_train, text_test = train_test_split(x, y, texts, test_size=test_size)

    return feature_columns, (x_train, y_train, text_train), (x_test, y_test, text_test)

# preprocess('./data/携程景点评价.csv')      