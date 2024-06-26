import torch
import torch.nn as nn
import json
import os
import numpy as np
from keras.src.models import Sequential
from fqdd.utils.argument import parse_arguments
from fqdd.utils.feature import get_feats
from keras.src.layers import Dense
from fqdd.utils.files import get_all_file, readtxt
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers
# from fqdd.utils.load_data import zero_pad_concat
from sklearn.model_selection import train_test_spit

args = parse_arguments()

dtexts = get_all_file("/data/work/own_learn/raw_data_2/dev", '.txt')
# ttexts = get_all_file("/data/work/own_learn/raw_data_2/test", '.txt')
dlabels = []
for f in dtexts:
    dlabels.append(readtxt(f))
# dlabels=['端木和于馨在一起了吗陈烁和云朵在一起了吗', '翻译我要去上海', '给公司打个电话', '看看我的日程表']
# print(dlabels)

dword = []
for item in dlabels:
    for it in item:
        dword.append(it)
print(dword)
dword2 = np.array(dword)
dword3 = dword2.reshape(1, -1)
dword4 = np.squeeze(dword3)
dword5 = list(set(dword4))
# print(dword5)
numdict = {i: w for i, w in enumerate(dword5)}
wdict = {w: i for i, w in enumerate(dword5)}
n_class = len(wdict)

n_step = 2  # number of steps ['宝 贝', '您 好']
n_hidden = 2  # number of hidden units


# 输入输出onr-hot化
def make_batch(sentences):
    input_data = []
    target_data = []
    # steps = 2
    for i, sen in enumerate(sentences):
        words = list(sen)  # 分字
        for j in range(len(words) - 2):
            tmp = []
            tmp.append(np.eye(n_class)[wdict[words[j]]])
            tmp.append(np.eye(n_class)[wdict[words[j]]])
            input_data.append(np.array(tmp))
            target_data.append(np.eye(n_class)[wdict[words[j + 2]]])
    return input_data, target_data


input_data, target_data = make_batch(dlabels)
input_data = np.array(input_data).reshape(-1, n_step * n_class)
target_data = np.array(target_data).reshape(-1, n_class)
X_train, X_test, y_train, y_test = train_test_spit(input_data, target_data, test_size=0.2, shuffle=False,
                                                    random_state=42)


# 模型
def define_model(optimizers):
    model = Sequential()

    model.add(Dense(512, activation='tanh', input_shape=(n_step * n_class,)))
    model.add(Dense(512, activation='tanh', input_shape=(512,)))
    model.add(Dense(n_class, activation='softmax'))  # 输出层
    model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy'], )
    model.summary()
    return model


optimizers = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
'''
Reduce = ReduceLROnPlateau(monitor='val_accuracy',
                         factor=0.1,
                         patience=5,
                         verbose=1,
                         mode='auto',
                         min_delta=0.0001,
                         cooldown=0,
                         min_lr=0)
'''
model = define_model(optimizers)
# model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[Reduce], verbose=2, epochs=5000)#训练5000轮，数据少啦，一两轮没效果
model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=2, epochs=5000)  # 训练5000轮，数据少啦，一两轮没效果

# 预测测试

predict = model.predict(X_test)
predict = np.argmax(predict, 1)  # 求取最大值索引

print('输入:', X_test)
print('预测：', predict)
