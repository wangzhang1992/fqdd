import librosa
import os
import sys
import wave
import uuid
import soundfile as sf
import random
import numpy as np
import time
source_wav_dir = "/data1/data_management/speech_processing/ASR/audio_raw/language_china/zh-CN/lable/near-field/read/SAD/Speaker_Independent/dataset/wav/train"

def randn_name():
    return str(uuid.uuid1()).replace("-", "") + str(time.time()).replace(".", "")[-5:]


def cut_wav(srcwav, starttime, endtime, sr=16000):
    waveform, sr = librosa.load(srcwav, sr=16000, mono=True)
    start_point = int(starttime * sr)
    end_point = int(endtime * sr)
    return waveform[start_point : end_point]


def combine_wave(selected_wavInfo, sr=16000):
    sents = []
    waveforms = None

    for info in selected_wavInfo:
        print(info)
        # wavpath = os.path.join(source_wav_dir, info["key"]+'.wav')
        wavpath = info["wavpath"]
        starttime = info["startTime"]
        endtime = info["endTime"]
        word = info["word"]
        sents.append(word)
        waveform_part = cut_wav(wavpath, starttime, endtime, sr)
        
        if type(waveforms) != np.ndarray:
            waveforms = waveform_part
        else:
            waveforms = np.hstack((waveforms, waveform_part))
    return "".join(sents), waveforms

def main():
    align_file = sys.argv[1] # key start end word
    result_dir = sys.argv[2]
    
    num_samples = 30000
    
    os.makedirs(result_dir, exist_ok=True)
    
    align_list = []
    with open(align_file, 'r', encoding='utf-8') as raf:
        for line in raf:
            lines = line.strip().split(" ")
            info = {"key": lines[0], "startTime": float(lines[1]), "endTime": float(lines[2]), "word": lines[3], "wavpath": lines[4]}
            align_list.append(info)
    times = 0
    with open(os.path.join(result_dir, "tran5.txt"), 'w', encoding='utf-8') as wt:
        while times < num_samples:
        
            num_words = random.randint(2,8)
            real_items = random.sample(align_list, num_words)
            sent, waveform = combine_wave(real_items, sr=16000)
            new_name = randn_name()
            wt.write("{} {}\n".format(new_name, sent))
            save_wav_path = os.path.join(result_dir, new_name+'.wav')
            sf.write(save_wav_path, waveform, samplerate=16000, format='wav')
            times += 1
if __name__ == "__main__":
    main()


