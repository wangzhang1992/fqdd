#!/usr/bin/python
# encoding: utf-8
import os, sys
import glob
import json



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


def prepare_data_list(data_folder, dirpath="data"):
    
    '''
    输出 data.list
    '''
    os.makedirs(dirpath, exist_ok=True)

    for dirname in ["train", "dev", "test"]:

        os.makedirs(os.path.join(dirpath, dirname), exist_ok=True) 
        datalist = os.path.join(dirpath, dirname, 'data.list')
        # /data1/data_management/speech_processing/ASR/audio_raw/language_china/zh-CN/lable/near-field/read/SAD/Speaker_Independent/dataset/{wav, transcript}
        trans_path = os.path.join(data_folder, "transcript", dirname+".txt") 
        if os.path.exists(trans_path):
            trans_map = readtxt(trans_path)
        else:
            trans_map = None

        train_wavs = glob.glob(os.path.join(data_folder, "wav", dirname)+'/*.wav')
        print(train_wavs[:10])
        train_wavs = [path for path in train_wavs if "channel" in path]
        print(train_wavs[:10])
        index = 0
        data_tmp = []
        for item in train_wavs:

            #if True:
            try:
                wavname = item.split("/")[-1].replace(".wav", "")
                key = wavname.split("_")[0]
                if trans_map:
                    trans = trans_map[key]
                    data_tmp.append({"id": index, "key": wavname, "wav":item, "txt":trans})
                else:
                    data_tmp.append({"id": index, "key": wavname, "wav":item, "txt": ""})
                index += 1
            except:
                continue
        with open(datalist, 'w', encoding="utf-8") as wf:
            for data in data_tmp:
                wf.write(json.dumps(data)+'\n')


def prepare_data(data_folder, dirpath="data"):

    '''
    输出 wav.scp text
    '''
    os.makedirs(dirpath, exist_ok=True)

    for dirname in ["train", "dev", "test"]:

        os.makedirs(os.path.join(dirpath, dirname), exist_ok=True)
        
        text_path = os.path.join(dirpath, dirname, "text")
        
        # /data1/data_management/speech_processing/ASR/audio_raw/language_china/zh-CN/lable/near-field/read/SAD/Speaker_Independent/dataset/{wav, transcript}
        trans_path = os.path.join(data_folder, "transcript", dirname+".txt")    
        wavpaths = glob.glob(os.path.join(data_folder, "wav", dirname)+'/*.wav')    
        # filter wav
        wavpaths = [path for path in wavpaths if "channel" in path]
        wavscp_path = os.path.join(dirpath, dirname, "wav.scp")
        
        if os.path.exists(trans_path):
            trans_map = readtxt(trans_path)
            with open(wavscp_path, 'w', encoding="utf-8") as wf, open(text_path, 'w', encoding='utf-8') as tf:
                for wavpath in wavpaths:

                    wavname = wavpath.split("/")[-1].replace(".wav", "")
                    key = wavname.split("_")[0]
                    wf.write(wavname+' '+wavpath +'\n')
                    tf.write(wavname+ " "+trans_map[key]+'\n')
        else:
            with open(wavscp_path, 'w', encoding="utf-8") as wf:
                for wavpath in wavpaths:
                    wavname = wavpath.split("/")[-1].replace(".wav", "")
                    wf.write(wavname+' '+wavpath +'\n')



# data_folder =  "/data1/data_management/speech_processing/ASR/audio_raw/language_china/zh-CN/lable/near-field/read/SAD/Speaker_Independent/dataset"

def main():
    
    data_folder = sys.argv[1]

    try:
        output_dir = sys.argv[2]
    except:
        output_dir = "data"

    prepare_data_list(data_folder, output_dir)
    prepare_data(data_folder, output_dir)

if __name__ == "__main__":
    main()
