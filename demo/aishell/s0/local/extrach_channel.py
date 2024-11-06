import torchaudio
import os, sys


wavscp = sys.argv[1]
newsample_rate = 16000

with open(wavscp, 'r') as rf:
    for line in rf:
        lines = line.strip().split(" ")
        name, wavpath = lines[0], lines[1]
        
        # 读取音频文件
        waveform, sample_rate = torchaudio.load(wavpath)
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=newsample_rate)(waveform)

        file_dir = os.path.dirname(wavpath)
        # 提取特定通道（例如，提取第一个通道）
        channels = waveform.shape[0]
        
        for channel in range(channels):
            channel_waveform = waveform[channel].unsqueeze(0)

            newpath = os.path.join(file_dir, name+"_channle"+str(channel)+".wav")
            # 保存提取的通道为新的音频文件
            torchaudio.save(newpath, channel_waveform, newsample_rate)
