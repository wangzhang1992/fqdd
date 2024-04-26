import sys
import numpy as np
import kaldi_io
from lang import readlexicon, words2phones


# TODO move batch processing to each model
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
    return labels


def convert(inputs, labels):
    xlen = [i.shape[0] for i in inputs]
    ylen = [i.shape[0] for i in labels]
    xs = zero_pad_concat(inputs)
    ys = end_pad_concat(labels)
    return xs, ys, xlen, ylen


class My_Data:
    def __init__(self, mode='train', batch_size=1, lexicon=None, phones=None):
        self.labels = {}
        self.feats_rspecifier = 'copy-feats scp:data/mfcc_pitch/{}/feats.scp ark:- | apply-cmvn ' \
                                '--utt2spk=ark:data/mfcc_pitch/{}/utt2spk scp:data/mfcc_pitch/{}/cmvn.scp ark:- ' \
                                'ark:- | add-deltas --delta-order=2 ark:- ark:- | nnet-forward ' \
                                'mfcc_pitch/final.feature_transform ark:- ark:- |'.format(mode, mode, mode)
        self.batch_size = batch_size
        # load label
        with open('data/mfcc_pitch/' + mode + '/text', 'r') as f:
            for line in f:
                line = line.strip().split()
                if lexicon is not None:
                    w2p = readlexicon(lexicon)
                    phone = words2phones(line[1], w2p, lexicon)
                    self.labels[line[0]] = np.array([phones[i] for i in phone])
                else:
                    self.labels[line[0]] = np.array([phones[i] for i in line[1]])

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        feats = []; label = []
        for k, v in kaldi_io.read_mat_ark(self.feats_rspecifier):
            if len(feats) >= self.batch_size:
                yield convert(feats, label)
                feats = []; label = []
            feats.append(v); label.append(self.labels[k])
        yield convert(feats, label)

'''
import editdistance


class TokenAcc():
    def __init__(self, blank=0):
        self.err = 0
        self.cnt = 0
        self.tmp_err = 0
        self.tmp_cnt = 0
        self.blank = 0
    
    def update(self, pred, xlen, label):
        #label is one dimensinal 
        pred = np.vstack([pred[i, :j] for i, j in enumerate(xlen)])
        e = self._distance(pred, label)
        c = label.shape[0]
        self.tmp_err += e; self.err += e
        self.tmp_cnt += c; self.cnt += c
        return 100 * e / c

    def get(self, err=True):
        # get interval
        if err: res = 100 * self.tmp_err / self.tmp_cnt
        else: res = 100 - 100 * self.tmp_err / self.tmp_cnt
        self.tmp_err = self.tmp_cnt = 0
        return res

    def getAll(self, err=True):
        if err: return 100 * self.err / self.cnt
        else: return 100 - 100 * self.err / self.cnt

    def _distance(self, y, t):
        if len(y.shape) > 1: 
            y = np.argmax(y, axis=1)
        print("y:{}".format(y))
        prev = self.blank
        hyp = []
        for i in y:
            if i != self.blank and i != prev: hyp.append(i)
            prev = i
        return editdistance.eval(hyp, t)
'''