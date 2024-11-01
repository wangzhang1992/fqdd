import os
import torch
import numpy as np
import torch.nn.functional as F
import Levenshtein

# args = parse_arguments()
# dic = dicts(args.train_path, args.test_path, args.dev_path)

'''
lists = [25,20,156,156,...,22,21]
dict ={' ':0,'a_1':2,...}
'''


def int2word(lists, dic):
    words = []
    for item in lists:
        for it in list(dic.keys()):
            if item == dic[it]:
                words.append(it)
    return words


'''
text:ai_1 b c a_2
dict ={' ':0,'a_1':2,...}
'''


def word2int(text, dic):
    ints = []
    for item in text:
        ints.append(dic[item])
    return ints


def GreedyDecoder(output, labels, wavlist_length, label_lengths, dic):
    # output = F.log_softmax(output, dim=2)
    output = output.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    arg_max = np.argmax(output, axis=2)
    
    wavlist_length = (wavlist_length * output.shape[1]).int()
    label_lengths = (label_lengths * labels.shape[1]).int()
    # print("arg_max.shape:{}".format(arg_max.shape)) # shape=(B, N)
    # print("labels.shape:{}".format(labels.shape)) # shape = (B, C)
    pred = [arg_max[i, :j] for i, j in enumerate(wavlist_length)]
    labels = [labels[i, :j] for i, j in enumerate(label_lengths)]
    decodes = []
    targets = []
    for i in range(wavlist_length.shape[0]):
        prev = 0
        pre = []
        targets.append(int2word(labels[i], dic))
        for j in pred[i]:
            if j != 0 and j !=1 and j !=2 and j != prev:
                pre.append(j)
            prev = j
        decodes.append(int2word(pre, dic))
    print("targets:{}\ndecodes:{}".format(targets, decodes))
    return targets, decodes


def decoder(output, dic):
    output = F.log_softmax(output, dim=2)
    output = output.squeeze(0).data.cpu().numpy()
    arg_maxes = np.argmax(output, axis=2)
    prev = 0
    pre = []
    for i in arg_maxes:
        if i != 0 and i != prev:
            pre.append(i)
        prev = i
    return ''.join(int2word(pre, dic))


def calculate_cer(targs, preds):
    if len(targs) == len(preds):
        cer = 0.
        for i, item in enumerate(targs):
            cer += Levenshtein.ratio(''.join(item), ''.join(preds[i]))
        return (1 - cer / len(targs)) * 100
    else:
        raise Exception("targs, preds not match")
        return 100.
