# encoding=utf8
import os, sys


def get_all_file(wavdir, tail):
    allfiles = []
    for root, dirs, files in os.walk(wavdir):
        for f in files:
            if f.endswith(tail):
                allfiles.append(os.path.join(root, f))

    return allfiles


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
    '''
    line:
        eg: S001T001E000N00011 感恩父母感恩亲人感恩全世界
    '''
    with open(txtf, 'r', encoding='utf8') as rf:
        str_map = {}
        for line in rf:
            try:
                key, value = line.strip().split(' ')
            except:
                key, value = line.strip().split('\t')
            finally:
                str_map[key] = value.lower()
        return str_map
# def readtxt(txtf):
#     with open(txtf, 'r', encoding='utf8') as rf:
#         strs = ''
#         for line in rf:
#             try:
#                 # strs = line.strip().split(' ')[1]
#                 strs = line.strip().replace(' ', '')
#             except:
#                 strs = line.strip().split('\t')[1].replace(' ', '')
#         return strs

# get_all_file('data/train/cctv','.wav')

# get_all_file('data/train/cctv','.wav')
