import os, sys
import pkuseg
import json
from script.utils.files import readtxt, get_all_file


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


def getdict_words(jsonpath):
    dicts = []
    try:
        jsonStr = json.load(open(jsonpath, 'r', encoding='utf-8'))
        for item in jsonStr:
            for word in jsonStr[item]["trans"]:
                if word in dicts:
                    pass
                else:
                    dicts.append(word)
        return dicts
    except:
        print("read jons file error,check: {}".format(jsonpath))
        return
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


def create_phones(dirpath):

    if not os.path.exists(os.path.join(dirpath, "phones.txt")): 
        try:
            print(os.path.join(dirpath, "train.json"))
            train = getdict_words(os.path.join(dirpath, "train.json"))
            test = getdict_words(os.path.join(dirpath, "test.json"))
            dev = getdict_words(os.path.join(dirpath, "dev.json"))
        except:
            print("data_json not exists")
            sys.exit()
        dicts = train + test + dev
        dicts = list(set(dicts))
        phones = {"<blank>":0, "<unk>": 1, "<sos/eos>":2}

        with open(os.path.join(dirpath, "phones.txt"), 'w', encoding='utf8') as ph:
            ph.write("<blank>\t0\n<unk>\t1\n<sos/eos>\t2\n")
            for idx in range(len(dicts)):
                ph.write(dicts[idx] + '\t' + str(idx + 3) + '\n')
                phones[dicts[idx]] = idx + 3
        return phones
    else:
        return read_phones(os.path.join(dirpath, "phones.txt"))


def read_phones(file_path):
    # lexicon and phones file format:
    # phones:zh\t1       words:中\t1
    # lexicon:中\tzh ong_1

    dic = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as rf:
            for line in rf:
                lines = line.strip().split('\t')
                dic[lines[0]] = int(lines[1])

    return dic
