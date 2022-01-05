import numpy as np
import argparse
from tqdm import tqdm
import logging
import tensorflow as tf
from tensorflow.python.keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn import metrics as sklearn_metrics

# from models.KD import A_Model, B_Model, Distilling
# 使用VGG注释上面即可
from models.VGG_KD import A_Model, B_Model, Distilling
from models.SA_CNN import SA_CNN

logging.getLogger().setLevel(logging.DEBUG)

def read_abstracts(path, label_num=2):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            filename, label1, label2 = line.strip().split()
            label = label2 if label_num==2 else label1

            yield f'./datasets/Abstracts/{filename}', int(label)

def read_mart(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            filename, value = line.split('\t')
            # 按照论文中提出的均值大于4为积极，虽然有些违和，但是为了和abstracts统一此处设定消极为1，积极为0
            label = abs(-(float(value) <= 4))

            yield f'./datasets/MART/{filename}', label

def read_data(path, name):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            filename, label = line.strip().split('\t')

            yield f'./datasets/{name}/{filename}', int(label)

def read_sub_data(path, name, p):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            filename, label = line.strip().split('\t')
            if int(label) != p:
                continue
            yield f'./datasets/{name}/{filename}', int(label)

def img_encode(img_paths):
    img_tensors = []
    for idx, ip in tqdm(enumerate(img_paths)):
        image_raw_data_jpg = tf.io.gfile.GFile(ip, 'rb').read()
        img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg, channels=3)
        img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)

        # 单通道转成3通道
        if tf.shape(img_data_jpg)[2] != 3:
            img_data_jpg = tf.image.grayscale_to_rgb(
                                img_data_jpg,
                                name=None
                            )
        '''
        method
        AREA	'area'
        BICUBIC	'bicubic'
        BILINEAR	'bilinear'
        GAUSSIAN	'gaussian'
        LANCZOS3	'lanczos3'
        LANCZOS5	'lanczos5'
        MITCHELLCUBIC	'mitchellcubic'
        NEAREST_NEIGHBOR	'nearest'
        '''
        # resize是通过算法缩放，通过method调整算法
        # img_tensors.append(tf.image.resize(img_data_jpg, [224, 224], method='bicubic'))
        # resize_with_crop_or_pad直接剪切，会损失很多
        img_tensors.append(tf.image.resize_with_crop_or_pad(img_data_jpg, 224, 224))
    return img_tensors

def k_fold(x_data, y_data, part=100, k=5):
    step = 100
    for i in range(0, len(x_data), step):
        yield np.array(x_data[:i]+x_data[i+part:]), y_data[:i]+y_data[i+part:], np.array(x_data[i:i+part]), y_data[i:i+part]

def train(config, encode_X, Y, mart_train_x, mart_train_y, mart_test_x, mart_test_y):
    # logging.debug(f'train: {len(train_x)}\ntest: {len(test_x)}')
    # train_x, test_x, train_y, test_y = train_test_split(np.array(encode_X), Y, test_size=0.3, random_state=seed)
    # mart_train_x, mart_test_x, mart_train_y, mart_test_y = train_test_split(np.array(mart_encode_X), MART_Y, test_size=0.2, random_state=seed)

    if config.model == 'KD':
        epoch_num = 1
        a_model = A_Model(len(set(Y)), np.array(encode_X), np.array(Y), epoch_num, 64).return_model()
        # eval(a_model, test_x, test_y, 64)
        b_model = B_Model().return_model()

        model = Distilling(b_model, a_model, 4, 0.9)
        model.compile(optimizer='adam',
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False))
        # callback = [tf.keras.callbacks.EarlyStopping(patience=5, monitor='loss')]
        callback = None

        distill_y = []
        for y in mart_train_y:
            one_hot = [1, 0] if y == 0 else [0, 1]
            distill_y.append(one_hot)
        model.fit(mart_train_x, np.array(distill_y), epochs=epoch_num, callbacks=callback)
        report = eval(model, mart_test_x, np.array(mart_test_y), 64)

    if config.model == 'SA_CNN':
        model = SA_CNN(len(set(mart_encode_X)))
        model.build_model()
        model.fit(mart_encode_X, np.array(mart_train_y), epochs=20, batch_size=32)

        report = model.evaluate(mart_test_x, np.array(mart_test_y), batch_size=32)
    
    return report


def eval(model, test_x, test_y, batch_size):
    y_pred = model.predict(test_x, batch_size=batch_size)
    y_pred = [np.argmax(onehot) for onehot in y_pred]
    report = sklearn_metrics.classification_report(test_y,
                                                   y_pred,
                                                   output_dict=True,
                                                   digits=4)
    for k, v in report.items():
        logging.debug(f'{k} {v}')
    cm = sklearn_metrics.confusion_matrix(test_y, y_pred)
    print(cm)

    return report


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="sentiment classify validation")
    parse.add_argument("--model", type=str, help='set model [KD, SA_CNN]', default='KD')
    ###改这
    parse.add_argument("--datasets", type=list, help='set datasets [MART, deviantArt, MART_neg, MART_pos, deviantArt_neg, deviantArt_pos]',
                         default=['MART','deviantArt'])
    parse.add_argument("--student_data", type=str, help='set datasets', default='MART')

    args = parse.parse_args()

    DATA, MART_DATA = [], []
    for dataset in args.datasets:
        if dataset == 'abstracts':
            for x, y in read_abstracts(f'./datasets/Abstracts/attrain1.txt~', label_num=2):
                DATA.append((x, y))
            for x, y in read_abstracts(f'./datasets/Abstracts/attest1.txt~', label_num=2):
                DATA.append((x, y))
        if dataset == 'MART':
            for x, y in read_data('./datasets/MART/scores.txt', 'MART'):
                DATA.append((x, y))
        if dataset == 'deviantArt':
            for x, y in read_data('./datasets/deviantArt/scores.txt', 'deviantArt'):
                DATA.append((x, y))
        if dataset == 'MART_neg':
            for x, y in read_sub_data('./datasets/MART/scores.txt', 'MART', 1):
                DATA.append((x, y))
        if dataset == 'MART_pos':
            for x, y in read_sub_data('./datasets/MART/scores.txt', 'MART', 0):
                DATA.append((x, y))
        if dataset == 'deviantArt_neg':
            for x, y in read_sub_data('./datasets/deviantArt/scores.txt', 'deviantArt', 1):
                DATA.append((x, y))
        if dataset == 'deviantArt_pos':
            for x, y in read_sub_data('./datasets/deviantArt/scores.txt', 'deviantArt', 0):
                DATA.append((x, y))
        
    if args.student_data == 'deviantArt':
        for x, y in read_data('./datasets/deviantArt/scores.txt', 'deviantArt'):
            MART_DATA.append((x, y))
    if args.student_data == 'MART':
        for x, y in read_data('./datasets/MART/scores.txt', 'MART'):
            MART_DATA.append((x, y))

    acc, p, r, f1 = [], [], [], []
    # [1, 11, 25, 35, 45, 65, 75, 100, 116, 154]
    # for shuffle_seed in [1]:
    shuffle_seed = 11
    np.random.seed(shuffle_seed)
    np.random.shuffle(DATA)
    np.random.shuffle(MART_DATA)
    encode_X = img_encode([data[0] for data in DATA])
    mart_encode_X = img_encode([data[0] for data in MART_DATA])
    Y = [data[1] for data in DATA]
    MART_Y = [data[1] for data in MART_DATA]
    print(len(Y), len(MART_Y))
    for mart_train_x, mart_train_y, mart_test_x, mart_test_y in k_fold(mart_encode_X, MART_Y, part=100, k=5):
        with tf.device("/gpu:0"):
            report = train(args, encode_X, Y, mart_train_x, mart_train_y, mart_test_x, mart_test_y)
        acc.append(report['accuracy'])
        p.append(report['macro avg']['precision'])
        r.append(report['macro avg']['recall'])
        f1.append(report['macro avg']['f1-score'])
    logging.debug(f'acc: {np.mean(acc)} p:{np.mean(p)} r:{np.mean(r)} f1:{np.mean(f1)}')
        # if np.mean(acc) > 0.785:
        #     with open('res.txt', 'a+', encoding='utf-8') as f:
        #         f.write(f'seed{shuffle_seed}, acc: {np.mean(acc)} p:{np.mean(p)} r:{np.mean(r)} f1:{np.mean(f1)}\n')
