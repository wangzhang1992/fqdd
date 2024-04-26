import torch
import torch.nn as nn
import json
import os
import torch.optim
import torch.utils.data as Data
import numpy as np
from keras.models import Sequential
from argument import parse_arguments
from feature import get_feats
from keras.layers import Dense
import keras
from files import get_all_file, readtxt
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from load_data import zero_pad_concat
from sklearn.model_selection import train_test_split

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
n_hidden = 512  # number of hidden units
m = 2


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
X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.2, shuffle=False,
                                                    random_state=42)
# 将数据装载到torch上
X_train = torch.LongTensor(X_train)
X_test = torch.LongTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

dataset = Data.TensorDataset(X_train, y_train)
loader = Data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)


# 定义网络结构，继承nn.Module
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        # 计算词向量表，大小是len(word_dict) * m
        self.C = nn.Embedding(n_class, m)

    def forward(self, x):
        '''
        X: [batch_size, n_step]
        '''
        # 根据词向量表，将我们的输入数据转换成三维数据
        # 将每个单词替换成相应的词向量
        x = self.C(x)  # [batch_size, n_step] => [batch_size, n_step, m]
        # 将替换后的词向量表的相同行进行拼接
        # view的第一个参数为-1表示自动判断需要合并成几行
        x = x.view(-1, n_step * m)  # [batch_size, n_step * m]
        self.tanh = torch.nn.functional.tanh(x)
        self.softmax = torch.nn.functional.softmax(self.tanh)
        return self.softmax
        # return output


model = NNLM()
# 分类问题用交叉熵作为损失函数
criterion = nn.CrossEntropyLoss()
# 优化器使用Adam
# 所谓的优化器，实际上就是你用什么方法去更新网路中的参数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 开始训练
for epoch in range(5000):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()

        # batch_x =batch_x.to('cuda')
        output = model(batch_x)

        # output : [batch_size, n_class], batch_y : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, batch_y)
        # 每1000次打印一次结果
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        # 反向传播更新梯度
        loss.backward()
        optimizer.step()

# Predict
# max()取的是最内层维度中最大的那个数的值和索引，[1]表示取索引
predict = model(X_test).data.max(1, keepdim=True)[1]

# Test
# squeeze()表示将数组中维度为1的维度去掉
# print([sen.split()[:n_step] for sen in sentences], '->', [numdict[n.item()] for n in predict.squeeze()])
print([numdict[n.item()] for n in predict.squeeze()])
