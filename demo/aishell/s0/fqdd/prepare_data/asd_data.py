import os, sys
import json
import glob
import torchaudio
from fqdd.utils.files import readtxt

# sys.path.insert(0, "./")




def prepare_data_list(data_folder, dirpath="data"):

    '''
    输出 data.list
    '''
    os.makedirs(dirpath, exist_ok=True)

    for dirname in ["train", "dev", "test"]:

        os.makedirs(os.path.join(dirpath, dirname), exist_ok=True)
        datalist = os.path.join(dirpath, dirname, 'data.list')
        if os.path.exists(datalist):
            continue
        # /data1/data_management/speech_processing/ASR/audio_raw/language_china/zh-CN/lable/near-field/read/SAD/Speaker_Independent/dataset/{wav, transcript}
        trans_path = os.path.join(data_folder, "transcript", dirname+".txt")
        if os.path.exists(trans_path):
            trans_map = readtxt(trans_path)
        else:
            trans_map = None
        train_wavs = glob.glob(os.path.join(data_folder, "wav", dirname)+'/*.wav')
        index = 0
        data_tmp = []
        for item in train_wavs:

            #if True:
            try:
                wavname = item.split("/")[-1].replace(".wav", "")
                if trans_map:
                    trans = trans_map[wavname]
                    data_tmp.append({"id": index, "key": wavname, "wav":item, "txt":trans})
                else:
                    data_tmp.append({"id": index, "key": wavname, "wav":item, "txt": ""})
                index += 1
            except:
                continue
        with open(datalist, 'w', encoding="utf-8") as wf:
            for data in data_tmp:
                wf.write(json.dumps(data)+'\n')


def prepare_data_json(data_folder, dirpath="data"):

    '''
    输出 data.list
    '''
    os.makedirs(dirpath, exist_ok=True)

    for dirname in ["train", "dev", "test"]:

        # os.makedirs(os.path.join(dirpath, dirname), exist_ok=True)
        os.makedirs(dirpath, exist_ok=True)
        datalist = os.path.join(dirpath, dirname+'.json')
        if os.path.exists(datalist):
            continue
        # /data1/data_management/speech_processing/ASR/audio_raw/language_china/zh-CN/lable/near-field/read/SAD/Speaker_Independent/dataset/{wav, transcript}
        trans_path = os.path.join(data_folder, "transcript", dirname+".txt")
        if os.path.exists(trans_path):
            trans_map = readtxt(trans_path)
        else:
            trans_map = None
        train_wavs = glob.glob(os.path.join(data_folder, "wav", dirname)+'/*.wav')
        index = 0
        data_tmp = {}
        for item in train_wavs:

            #if True:
            try:
                wav_info = torchaudio.info(item)
                duration = wav_info.num_frames / wav_info.sample_rate
                wavname = item.split("/")[-1].replace(".wav", "")
                if trans_map:
                    trans = trans_map[wavname]
                    data_tmp[index] = {"key": wavname, "path":item, "trans":trans, "during": duration}
                else:
                    data_tmp[index] = {"key": wavname, "path":item, "txt": "", "during": duration}
                index += 1
            except Exception as e:
                print(e)
                continue
        js = json.dumps(data_tmp, indent=2, ensure_ascii=False)
        print(datalist)
        with open(datalist, "w", encoding="utf-8") as wf:
            wf.write(js)

def prepare_data(data_folder, dirpath="data"):

    '''
    输出 wav.scp text
    '''
    os.makedirs(dirpath, exist_ok=True)

    for dirname in ["train", "dev", "test"]:

        os.makedirs(os.path.join(dirpath, dirname), exist_ok=True)

        text_path = os.path.join(dirpath, dirname, "text")
        if os.path.exists(text_path):
            pass
        else:
            # /data1/data_management/speech_processing/ASR/audio_raw/language_china/zh-CN/lable/near-field/read/SAD/Speaker_Independent/dataset/{wav, transcript}
            trans_path = os.path.join(data_folder, "transcript", dirname+".txt")

            if os.path.exists(trans_path):
                trans_map = readtxt(trans_path)

                with open(text_path, 'w', encoding='utf-8') as wf:
                    for key, value in trans_map.items():
                        wf.write(key+ " "+value+'\n')

        wavpaths = glob.glob(os.path.join(data_folder, "wav", dirname)+'/*.wav')
        wavscp_path = os.path.join(dirpath, dirname, "wav.scp")
        if os.path.exists(wavscp_path):
            pass
        else:
            with open(wavscp_path, 'w', encoding="utf-8") as wf:
                for wavpath in wavpaths:
                    wavname = wavpath.split("/")[-1].replace(".wav", "")
                    wf.write(wavname+' '+wavpath +'\n')


'''
data_folder =  "/data1/data_management/speech_processing/ASR/audio_raw/language_china/zh-CN/lable/near-field/read/SAD/Speaker_Independent/dataset"
prepare_data_json(data_folder, "data")
'''
