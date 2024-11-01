import os, sys
import glob
import json
#sys.path.insert(0, "./")
from fqdd.utils.files import get_all_file, readtxt
from fqdd.utils.dataio import read_audio


def prepare_data(data_folder, dirpath):

    os.makedirs(dirpath, exist_ok=True)
    
    for dirname in ["train", "dev", "test"]:
        if os.path.exists(os.path.join(dirpath, dirname+'.json')):
            continue
        train_wavs = glob.glob(os.path.join(data_folder, dirname)+'/*/*.wav')
        index = 0
        tmp = {}
        for item in train_wavs:
            
            #if True:
            try:
                trans = readtxt(item.replace('.wav', '.txt'))
                tmp[index] = {"path":item, "during": read_audio(item).shape[0]/16000, "trans":trans}
                index += 1
            except:
                continue
        js = json.dumps(tmp, indent=2, ensure_ascii=False)
        print(os.path.join(dirpath, dirname+'.json'))
        with open(os.path.join(dirpath, dirname+'.json'), "w", encoding="utf-8") as wf:
            wf.write(js)


# prepare_data("test_folder", "result/2021")
