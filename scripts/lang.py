import os, sys
import pkuseg
from files import readtxt, get_all_file


def readlexicon(lexicon):
    w2p = {}
    with open(lexicon, 'r', encoding='utf-8') as rf:
        for line in rf:
            item = line.strip().split('\t')
            w2p[item[0]] = item[1]
    return w2p


def words2phones(sent, w2p, lexicon):
    pk = pkuseg.pkuseg(user_dict=lexicon)
    phones = ''
    if sent == '':
        return phones
    else:
        words = ' '.join(pk.cut(sent))
        for word in words:
            if word in list(w2p.keys()):
                phones += w2p[word].strip() + ' '
            else:
                for item in word:
                    if item in list(w2p.keys()):
                        phones += w2p[word].strip() + ' '
                    else:
                        phones += "<UNK> "
    phones = phones.strip().split(' ')
    return phones


def getdict_words(txtpath):
    dicts = []
    txtfile = get_all_file(txtpath, 'txt')
    for f in txtfile:
        txt = readtxt(f)
        for word in txt.strip():
            if word in dicts:
                pass
            else:
                dicts.append(word)
    return dicts


def getdict_phones(txtpath, w2p, lexicon):
    dicts = []
    txtfile = get_all_file(txtpath, 'txt')
    for f in txtfile:
        txt = readtxt(f)
        phones = words2phones(txt, w2p, lexicon)
        for phone in phones:
            if phone not in dicts:
                dicts.append(phone)
    return dicts


def dicts(train_path, test_path, dev_path, lexicon=None):
    if lexicon is None:
        train = getdict_words(train_path)
        test = getdict_words(test_path)
        dev = getdict_words(dev_path)
    else:
        w2p = readlexicon(lexicon)
        train = getdict_phones(train_path, w2p, lexicon)
        test = getdict_phones(test_path, w2p, lexicon)
        dev = getdict_phones(dev_path, w2p, lexicon)
    dicts = train + test + dev
    dicts = list(set(dicts))
    dics = {"<ese>": 0}
    with open('data/phones.txt', 'w', encoding='utf8') as dic:
        dic.write("<ese> 0\n")
        for idx in range(len(dicts)):
            dic.write(dicts[idx] + ' ' + str(idx + 1) + '\n')
            dics[dicts[idx]] = idx + 1
    return dics

# dicts(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
