import glob
import json
import os

from fqdd.utils.dataio import read_audio


def prepare_data(data_folder, dirpath):

    os.makedirs(dirpath, exist_ok=True)
    text_map = {}
    text_path = os.path.join(data_folder, "transcript/aishell_transcript_v0.8.txt")
    if os.path.exists(text_path):
        with open(text_path, 'r', encoding='utf-8') as rf:
            for item in rf:
                items = item.strip().split(' ')
                if len(items)>=2:
                    text_map[items[0]] = " ".join(items[1:]).replace(" ","")
    
    for dirname in ["train", "dev", "test"]:
        if os.path.exists(os.path.join(dirpath, dirname+'.json')):
            continue
        train_wavs = glob.glob(os.path.join(data_folder, 'wav', dirname)+'/*/*.wav')
        index = 0
        tmp = {}
        for item in train_wavs:
            
            #if True:
            try:
                ids = item.split("/")[-1].split(".")[0]
                tmp[index] = {"path":item, "during": read_audio(item).shape[0]/16000, "trans":text_map[ids]}
                index += 1

            except:
                continue
        js = json.dumps(tmp, indent=2, ensure_ascii=False)
        print(os.path.join(dirpath, dirname+'.json'))
        with open(os.path.join(dirpath, dirname+'.json'), "w", encoding="utf-8") as wf:
            wf.write(js)


# prepare_data("/data1/data_management/speech_processing/ASR/audio_raw/language_china/zh-CN/lable/near-field/read/aishell/aishell_1_178hr/data_aishell/", "result/2021")
