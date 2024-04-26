import os
import numpy as np
import torch.autograd.variable as Variable
import torch.nn as nn
import torch


def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t) + inputs[0].shape[1:]
    input_mat = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0]] = inp
    return input_mat


def end_pad_concat(inputs):
    max_t = max(i.shape[0] for i in inputs)
    shape = (len(inputs), max_t)
    labels = np.full(shape, fill_value=inputs[0][-1], dtype='i')

    for e, l in enumerate(inputs):
        labels[e, :len(l)] = l
    print(labels)
    return labels


def convert(inputs, labels):
    xlen = [i.shape[0] for i in inputs]
    ylen = [i.shape[0] for i in labels]
    xs = zero_pad_concat(inputs)
    ys = end_pad_concat(labels)
    return xs, ys, xlen, ylen


input = []
labels = []
np.random.seed(2021)
for i in np.random.randint(50,500,2):
    input.append(np.random.rand(i,20))

for i in np.random.randint(2,20,2):
    labels.append(np.random.randint(1,20,i))

xs, ys, xlen, ylen = convert(input, labels)
x = torch.FloatTensor(xs)
print(ys)
ys = np.hstack([ys[i, :j] for i, j in enumerate(ylen)])
y = torch.IntTensor(ys)
x1 = torch.IntTensor(xlen)
y1 = torch.IntTensor(ylen)
print(y)
print(x.size())
