import os
import librosa
import torchaudio
import python_speech_features
import numpy as np
import scipy.io as scio
import torchaudio
import scipy.io.wavfile as wav
from fqdd.utils.files import get_all_file, readtxt



def resample_waveform(waveform, org_fr, new_fr):
    waveform = torchaudio.transforms.Resample(orig_freq=org_fr, new_freq=new_fr)(waveform)
    return waveform

def extract_feat(wavpath, sr=16000, feat_type='mfcc', feat_cof=40, lowfreq=0):
    waveform, sample_rate = torchaudio.load(wavpath)  
    waveform = waveform * (1 << 15) 
    if sr == sample_rate:
        pass
    else:
        waveform = resample_waveform(waveform, sample_rate, sr)
    
    if feat_type =='raw':
        feat = waveform.squeeze()
    else:
        extract_fn = torchaudio.compliance.kaldi.fbank if feat_type == "fbank" else torchaudio.compliance.kaldi.mfcc
        feat =  extract_fn(waveform, num_mel_bins=feat_cof, channel=-1, sample_frequency=sample_rate)
   
    # extract pitch
    # feat =  np.concatenate([feat, pitch], axis=0)
    return feat


def get_feats(wavdir, save_path, feat_type='mfcc', feat_cof=40, min_during=1, max_during=15):
    os.makedirs(save_path, exist_ok=True)
    if os.path.exists(save_path + '/feat.mat') and os.path.exists(save_path + '/label.mat'):
        print('File Exists:', save_path + '/feat.mat', save_path + '/label.mat')
        return True
    wavs = get_all_file(wavdir, '.wav')
    feats = []
    labels = []
    save_feat = dict()
    save_label = dict()
    np.random.shuffle(wavs)
    for i, wav_path in enumerate(wavs):
        label_path = wav_path.replace('.wav', '.txt')
        file_name = wav_path.split('/')[-1].replace('.wav', '')
        if os.path.exists(wav_path) and os.path.exists(label_path):
            x, sr = librosa.load(wavs[i], 16000)
            # print(librosa.get_duration(x,sr))
            if librosa.get_duration(x, sr) < min_during or librosa.get_duration(x, sr) > max_during:
                continue
            feat = extract_feat(x, sr, feat_type=feat_type, feat_cof=feat_cof)  # (data, time)
            # if feat_type=='mfcc':
            # feat = np.transpose(feat, (1, 0))  # (time, data)
            # print(feat.shape[0])
            feats.append(feat)
            try:
                label = readtxt(label_path)
                labels.append(label)
            except:
                print("label file not exist")
                continue
            save_feat[file_name] = feat
            save_label[file_name] = label

    scio.savemat(save_path + '/feat.mat', save_feat)
    scio.savemat(save_path + '/label.mat', save_label)
    return True


# extract_feat(sys.argv[1])
'''
get_feats('/data/work/own_learn/raw_data/dev/', "/data/work/own_learn/data/dev/", feat_type='mfcc', feat_cof=40)
get_feats('/data/work/own_learn/raw_data/test/', "/data/work/own_learn/data/test/", feat_type='mfcc', feat_cof=40)
get_feats('/data/work/own_learn/raw_data/train/', "/data/work/own_learn/data/train/", feat_type='mfcc', feat_cof=40)

feat_mat = '/data/work/own_learn/data/dev/feat.mat'
label_mat = '/data/work/own_learn/data/dev/label.mat'
feats = scio.loadmat(feat_mat)
labels = scio.loadmat(label_mat)

for key, value in labels.items():
    if key in ['__header__','__version__','__globals__']:
        continue
    print(feats[key])
'''
