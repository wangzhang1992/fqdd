import os, sys
import simplejson
import pkuseg


def isExitesFile(filePath):
    filePath = filePath.strip()
    isExists = os.path.exists(filePath)
    if not isExists:
        os.makedirs(filePath)
        print(filePath + ' create success')
        return True
    else:
        print(filePath + ' dir exites')
        return False


def clean_txt(txt):
    dots = ['·', '「', ' ', '	', '\\', ',', '.', '/', '~', '!', '#', '$', '%', '^', '&', '*', '(', ')', '_', '-', '=',
            '+', '[', ']', '|', ';', ':', '\'', '"', '?', '>', '<', '，', '。', ';', '‘', '’', '·', '「', '{', '}', '|',
            '“', '”', '：', '》', '《', '？', '～', '@', '￥', '（', '）', '——', '」', '`', '！']
    if txt is '':
        return ''
    else:
        for dot in dots:
            txt = txt.replace(dot, '')
        return txt


def get_all_file(rawdir, tail):
    allfile = []
    for root, dirs, files in os.walk(rawdir):
        for f in files:
            if f.endswith(tail):
                allfile.append(os.path.join(root, f))
    return allfile


def json2txt(json_txt, store_path):
    jsonf = open(json_txt, 'r', encoding='utf8')
    jsons = simplejson.load(jsonf)
    for item in range(len(jsons)):
        txt = jsons[item]['text'].strip()
        txt = clean_txt(txt)
        ids = jsons[item]['id'].strip()
        user_id = jsons[item]['user_id'].strip()
        wav_name = jsons[item]['file'].strip().split('.')[0]
        with open(store_path, 'a', encoding='utf8') as wf:
            wf.write(wav_name + '\t' + ids + '\t' + user_id + '\t' + txt + '\n')


def readf(f):
    file_read = open(f.replace('.wav', '.txt'), 'r', encoding='utf8')
    strs = ''
    for line in file_read:
        if line.strip() == '':
            strs = strs + ''
        else:
            # strs = strs + line.strip().split('\t')[1]
            strs = strs + line.strip()
    return strs


def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            pass
        else:
            return False
    return True


def main():
    data_path = sys.argv[1]
    dis_path = sys.argv[2]
    files = get_all_file(data_path, '.wav')
    isExitesFile(dis_path)
    wavscp = open(dis_path + '/wav.scp', 'w', encoding='utf8')
    utt2spk = open(dis_path + '/utt2spk', 'w', encoding='utf8')
    word = open(dis_path + '/text', 'w', encoding='utf8')
    # pk = pkuseg.pkuseg(user_dict='data/dict/dict.txt')
    for f in files:
        if os.path.exists(f.replace('.wav', '.txt')) is False:
            continue
        if os.path.getsize(f) <= 44:
            continue
        print(f)
        strs = clean_txt(readf(f))
        if strs == '':
            continue
        # if not is_Chinese(strs):
        #    continue
        wav_name = f.split('/')[-1].split('.')[0]
        wavscp.write(wav_name + '\t' + f + '\n')
        # word.write(wav_name+'\t'+' '.join(pk.cut(strs))+'\n')
        word.write(wav_name + '\t' + strs + '\n')
        utt2spk.write(wav_name + ' ' + wav_name.split('_')[0] + '\n')
    wavscp.close()
    utt2spk.close()
    word.close()


if __name__ == "__main__":
    main()
