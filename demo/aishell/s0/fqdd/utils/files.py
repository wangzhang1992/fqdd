# encoding=utf8
import os, sys


def get_all_file(wavdir, tail):
    allfiles = []
    for root, dirs, files in os.walk(wavdir):
        for f in files:
            if f.endswith(tail):
                allfiles.append(os.path.join(root, f))

    return allfiles


def readtxt(txtf):
    with open(txtf, 'r', encoding='utf8') as rf:
        strs = ''
        for line in rf:
            try:
                # strs = line.strip().split(' ')[1]
                strs = line.strip().replace(' ', '')
            except:
                strs = line.strip().split('\t')[1].replace(' ', '')
        return strs

# get_all_file('data/train/cctv','.wav')
