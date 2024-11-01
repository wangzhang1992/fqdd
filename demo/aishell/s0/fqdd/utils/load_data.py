import copy
import os, sys
import random
import scipy
import torch
import json
import math
import logging
import torchaudio
import numpy as np

# sys.path.insert(0, "./")
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DistributedSampler, DataLoader
from fqdd.utils.init_tokenizer import Tokenizers

# '''
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


# '''

class Dataload:
    def __init__(
            self,
            filelist,
            conf=None,
            tokenizer=None,
    ):

        self.filelist = filelist
        self.files = [json.loads(line.strip()) for line in open(filelist, 'r').readlines()]

        self.tokenizer = tokenizer

        if conf.get("filter", False):
            # logging.info("data_list_len_before:{}".format(len(self.files)))
            self.files = self.filter(conf["filter_conf"])
            # logging.info("data_list_len_after:{}".format(len(self.files)))
            self.files = self.sortD(reverse=True)

        # logging.info("data_list_len:{}".format(len(self.files)))
        self.conf = conf

        # feat_conf=conf["feat_conf"]
        # augment_conf=conf["augment"]

    def speed_perturb(self, waveform, sr, speeds=None):
        """ Apply speed perturb to the sample.
            Inplace operation.

            Args:
                waveform: torch.FloatTensor
                speeds(List[float]): optional speed

            Returns:
                key, wav, label, sample_rate}
        """

        if speeds is None:
            speeds = [0.9, 1.0, 1.1]
        speed = random.choice(speeds)
        if speed != 1.0:
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sr,
                [['speed', str(speed)], ['rate', str(sr)]])

        return waveform

    def wav_distortion(self, waveform, distort_conf, distort_types=None):

        def db2amp(db):
            return pow(10, db / 20)

        def amp2db(amp):
            return 20 * math.log10(amp)

        def make_poly_distortion(conf):
            """Generate a db-domain ploynomial distortion function

                f(x) = a * x^m * (1-x)^n + x

            Args:
                conf: a dict {'a': #int, 'm': #int, 'n': #int}

            Returns:
                The ploynomial function, which could be applied on
                a float amplitude value
            """
            a = conf['a']
            m = conf['m']
            n = conf['n']

            def poly_distortion(x):
                abs_x = abs(x)
                if abs_x < 0.000001:
                    x = x
                else:
                    db_norm = amp2db(abs_x) / 100 + 1
                    if db_norm < 0:
                        db_norm = 0
                    db_norm = a * pow(db_norm, m) * pow((1 - db_norm), n) + db_norm
                    if db_norm > 1:
                        db_norm = 1
                    db = (db_norm - 1) * 100
                    amp = db2amp(db)
                    if amp >= 0.9997:
                        amp = 0.9997
                    if x > 0:
                        x = amp
                    else:
                        x = -amp
                return x

            return poly_distortion

        def make_quad_distortion(conf):
            return make_poly_distortion({"a": conf.get("a", 1), "m": conf.get("m", 1), "n": conf.get("n", 1)})

        # the amplitude are set to max for all non-zero point
        def make_max_distortion(conf):
            """Generate a max distortion function

            Args:
                conf: a dict {'max_db': float }
                    'max_db': the maxium value.

            Returns:
                The max function, which could be applied on
                a float amplitude value
            """
            max_db = conf['max_db']
            if max_db:
                max_amp = db2amp(max_db)  # < 0.997
            else:
                max_amp = 0.997

            def max_distortion(x):
                if x > 0:
                    x = max_amp
                elif x < 0:
                    x = -max_amp
                else:
                    x = 0.0
                return x

            return max_distortion

        def make_amp_mask(db_mask=None):
            """Get a amplitude domain mask from db domain mask

            Args:
                db_mask: Optional. A list of tuple. if None, using default value.

            Returns:
                A list of tuple. The amplitude domain mask
            """
            if db_mask is None:
                db_mask = [(-110, -95), (-90, -80), (-65, -60), (-50, -30), (-15, 0)]
            amp_mask = [(db2amp(db[0]), db2amp(db[1])) for db in db_mask]
            return amp_mask

        default_mask = make_amp_mask()

        def generate_amp_mask(mask_num):
            """Generate amplitude domain mask randomly in [-100db, 0db]

            Args:
                mask_num: the slot number of the mask

            Returns:
                A list of tuple. each tuple defines a slot.
                e.g. [(-100, -80), (-65, -60), (-50, -30), (-15, 0)]
                for #mask_num = 4
            """
            a = [0] * 2 * mask_num
            a[0] = 0
            m = []
            for i in range(1, 2 * mask_num):
                a[i] = a[i - 1] + random.uniform(0.5, 1)
            max_val = a[2 * mask_num - 1]
            for i in range(0, mask_num):
                l = ((a[2 * i] - max_val) / max_val) * 100
                r = ((a[2 * i + 1] - max_val) / max_val) * 100
                m.append((l, r))
            return make_amp_mask(m)

        def make_fence_distortion(conf):
            """Generate a fence distortion function

            In this fence-like shape function, the values in mask slots are
            set to maxium, while the values not in mask slots are set to 0.
            Use seperated masks for Positive and negetive amplitude.

            Args:
                conf: a dict {'mask_number': int,'max_db': float }
                    'mask_number': the slot number in mask.
                    'max_db': the maxium value.

            Returns:
                The fence function, which could be applied on
                a float amplitude value
            """
            mask_number = conf['mask_number']
            max_db = conf['max_db']
            max_amp = db2amp(max_db)  # 0.997
            if mask_number <= 0:
                positive_mask = default_mask
                negative_mask = make_amp_mask([(-50, 0)])
            else:
                positive_mask = generate_amp_mask(mask_number)
                negative_mask = generate_amp_mask(mask_number)

            def fence_distortion(x):
                is_in_mask = False
                if x > 0:
                    for mask in positive_mask:
                        if x >= mask[0] and x <= mask[1]:
                            is_in_mask = True
                            return max_amp
                    if not is_in_mask:
                        return 0.0
                elif x < 0:
                    abs_x = abs(x)
                    for mask in negative_mask:
                        if abs_x >= mask[0] and abs_x <= mask[1]:
                            is_in_mask = True
                            return max_amp
                    if not is_in_mask:
                        return 0.0
                return x

            return fence_distortion

        #
        def make_jag_distortion(conf):
            """Generate a jag distortion function

            In this jag-like shape function, the values in mask slots are
            not changed, while the values not in mask slots are set to 0.
            Use seperated masks for Positive and negetive amplitude.

            Args:
                conf: a dict {'mask_number': #int}
                    'mask_number': the slot number in mask.

            Returns:
                The jag function,which could be applied on
                a float amplitude value
            """
            mask_number = conf['mask_number']
            if mask_number <= 0:
                positive_mask = default_mask
                negative_mask = make_amp_mask([(-50, 0)])
            else:
                positive_mask = generate_amp_mask(mask_number)
                negative_mask = generate_amp_mask(mask_number)

            def jag_distortion(x):
                is_in_mask = False
                if x > 0:
                    for mask in positive_mask:
                        if x >= mask[0] and x <= mask[1]:
                            is_in_mask = True
                            return x
                    if not is_in_mask:
                        return 0.0
                elif x < 0:
                    abs_x = abs(x)
                    for mask in negative_mask:
                        if abs_x >= mask[0] and abs_x <= mask[1]:
                            is_in_mask = True
                            return x
                    if not is_in_mask:
                        return 0.0
                return x

            return jag_distortion

        # gaining 20db means amp = amp * 10
        # gaining -20db means amp = amp / 10
        def make_gain_db(conf):
            """Generate a db domain gain function

            Args:
                conf: a dict {'db': #float}
                    'db': the gaining value

            Returns:
                The db gain function, which could be applied on
                a float amplitude value
            """
            db = conf['db']

            def gain_db(x):
                return min(0.997, x * pow(10, db / 20))

            return gain_db

        def distort(x, func, rate=0.8):
            """Distort a waveform in sample point level

            Args:
                x: the origin wavefrom
                func: the distort function
                rate: sample point-level distort probability

            Returns:
                the distorted waveform
            """
            for i in range(0, x.shape[1]):
                a = random.uniform(0, 1)
                if a < rate:
                    x[0][i] = func(float(x[0][i]))
            # logging.info("distort_after:{}".format(type(x)))
            return x

        # x is numpy

        waveform = waveform.detach().numpy()
        rate = distort_conf["rate"]

        if distort_types is None:
            distort_types = ['gain_db', 'max_distortion', 'fence_distortion', 'jag_distortion', 'poly_distortion',
                             'quad_distortion', 'none_distortion']
        distort_type = random.choice(distort_types)

        if distort_type == 'gain_db':
            gain_db = make_gain_db(distort_conf["gain_db"])
            waveform = distort(waveform, gain_db)
        elif distort_type == 'max_distortion':
            max_distortion = make_max_distortion(distort_conf["max_distortion"])
            waveform = distort(waveform, max_distortion, rate=rate)
        elif distort_type == 'fence_distortion':
            fence_distortion = make_fence_distortion(distort_conf["fence_distortion"])
            waveform = distort(waveform, fence_distortion, rate=rate)
        elif distort_type == 'jag_distortion':
            jag_distortion = make_jag_distortion(distort_conf["jag_distortion"])
            waveform = distort(waveform, jag_distortion, rate=rate)
        elif distort_type == 'poly_distortion':
            poly_distortion = make_poly_distortion(distort_conf["poly_distortion"])
            waveform = distort(waveform, poly_distortion, rate=rate)
        elif distort_type == 'quad_distortion':
            quad_distortion = make_quad_distortion(distort_conf["quad_distortion"])
            waveform = distort(waveform, quad_distortion, rate=rate)
        elif distort_type == 'none_distortion':
            pass
        else:
            logging.info('unsupport type')

        '''
        if type(waveform) != np.ndarray:
            logging.info("***************************************\ndistort_type:{}".format(distort_type))
            logging.info(waveform)
        '''
        return torch.from_numpy(waveform)

    def readwav(self, wavfile, start: float = 0.0, end: float = 0.0):

        sr = torchaudio.info(wavfile).sample_rate
        if start != 0.0:
            assert end != 0.0
            start_frame = int(sr * start)
            end_frame = int(sr * end)

            waveform, _ = torchaudio.load(
                wavfile,
                num_frames=end_frame - start_frame,
                frame_offset=start_frame
            )
        else:
            waveform, _ = torchaudio.load(wavfile)
        return waveform, sr  # size = (1, t), 16000

    def add_noise(self, waveform, add_noise_conf, resample_rate=16000):

        noise_lists = add_noise_conf.get("noise_lists", None)
        snr_dbs = add_noise_conf.get("snr_db", [5, 10, 15])
        rate = add_noise_conf.get("rate", 0.0)

        noise_paths = [line.strip() for line in open(noise_lists, 'r').readlines()]

        if len(noise_paths) > 0 and rate < random.uniform(0, 1):
            snr_db = random.choice(snr_dbs)
            n_waveform, n_sr = torchaudio.load(random.choice(noise_paths))
            n_waveform = torchaudio.transforms.Resample(
                orig_freq=n_sr,
                new_freq=resample_rate
            )(n_waveform)

            # 确保噪音长度至少和语音长度一样长
            if n_waveform.size(1) < waveform.size(1):
                n_waveform = torch.cat([n_waveform] * (waveform.size(1) // n_waveform.size(1) + 1), dim=1)
            n_waveform = n_waveform[:, :waveform.size(1)]

            audio_power = waveform.norm(p=2)
            noise_power = n_waveform.norm(p=2)
            snr = 10 ** (snr_db / 20)
            scale = snr * noise_power / audio_power
            waveform = (waveform + scale * n_waveform) / 2
        return waveform

    def add_reverb(self, waveform, add_reverb_conf, resample_rate=16000):

        reverb_lists = add_reverb_conf.get("reverb_lists", None)
        rate = add_reverb_conf.get("rate", 0.0)

        reverb_paths = [line.strip() for line in open(reverb_lists, 'r').readlines()]

        if len(reverb_paths) > 0 and rate < random.uniform(0, 1):

            r_waveform, r_sr = torchaudio.load(random.choice(reverb_paths))
            r_waveform = torchaudio.transforms.Resample(
                orig_freq=r_sr,
                new_freq=resample_rate
            )(r_waveform)
            # 确保音频和脉冲响应是单声道
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0).unsqueeze(0)
            if r_waveform.size(0) > 1:
                r_waveform = r_waveform.mean(dim=0).unsqueeze(0)

            '''
            使用 scipy 的 fftconvolve 进行卷积
            full: 默认模式。
                计算输入信号和卷积核的完全卷积，输出的大小为 𝑁+𝑀−1,其中 N 是输入信号的长度，M 是卷积核的长度。
                包含所有部分重叠和边缘效应。

            valid: 只保留完全重叠部分，忽略边缘效应。
                输出的大小为 𝑁−𝑀+1（假设 𝑁≥𝑀。仅在输入信号完全包含卷积核的范围内计算结果。

            same:输出大小与输入信号相同。
                在中心部分进行完全重叠卷积，并根据需要用零填充边缘来匹配输入信号的大小。输入和输出信号长度相同
            '''
            reverb_waveform = scipy.signal.fftconvolve(waveform.numpy(), r_waveform.numpy(), mode='full')
            # 进行标准化以防止溢出
            reverb_waveform = reverb_waveform / np.max(np.abs(reverb_waveform))
            waveform = reverb_waveform[:, :waveform.shape[1]]
            waveform = torch.from_numpy(waveform)
            # logging.info(waveform.shape)
        return waveform

    def compute_feat(self, waveform):

        waveform = waveform * (1 << 15)

        feat_conf = self.conf["feat_conf"]
        sample_rate = self.conf.get("sample_rate", 16000)

        if self.conf.get("feat_type") == "fbank":
            mat = torchaudio.compliance.kaldi.fbank(
                waveform,
                num_mel_bins=feat_conf.get("num_mel_bins", 80),
                frame_length=feat_conf.get("frame_length", 25),
                frame_shift=feat_conf.get("frame_shift", 10),
                dither=feat_conf.get("dither", 0.0),
                energy_floor=feat_conf.get("energy_floor", 0.0),
                sample_frequency=sample_rate
            )
        elif self.conf.get("feat_type") == "mfcc":
            mat = torchaudio.compliance.kaldi.mfcc(
                waveform,
                num_mel_bins=feat_conf.get("num_mel_bins", 23),
                frame_length=feat_conf.get("frame_length", 25),
                frame_shift=feat_conf.get("frame_shift", 10),
                dither=feat_conf.get("dither", 0.0),
                num_ceps=feat_conf.get("num_ceps", 40),
                high_freq=feat_conf.get("high_freq", 0.0),
                low_freq=feat_conf.get("low_freq", 0.0),
                sample_frequency=sample_rate
            )

        else:
            mat = waveform
        return mat

    def spec_aug(self, feat, spec_aug_conf):
        """
        function:
            Do spec augmentation
            Inplace operation

        Args:
            feat: torch.FloatTensor
            spec_aug_conf

        Returns
            feat: torch.FloatTensor
        """

        assert isinstance(feat, torch.Tensor)

        num_t_mask = spec_aug_conf.get("num_t_mask", 2)
        num_f_mask = spec_aug_conf.get("num_f_mask", 2)
        max_t = spec_aug_conf.get("max_t", 50)
        max_f = spec_aug_conf.get("max_f", 10)
        max_w = spec_aug_conf.get("max_w", 80)
        rate = spec_aug_conf.get("rate", 0)

        if rate < random.uniform(0, 1):

            y = feat.clone().detach()
            max_frames = y.size(0)
            max_freq = y.size(1)
            # time mask
            for i in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                y[start:end, :] = 0
            # freq mask
            for _ in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                y[:, start:end] = 0
            return y
        else:
            return feat

    def spec_sub(self, feat, spec_sub_conf):
        """
        funciton:
            Do spec substitute
            Inplace operation
            ref: U2++, section 3.2.3 [https://arxiv.org/abs/2106.05642]

        Args:
            feat:  torch.FloatTensor
            spec_sub_conf

        Returns
            feat: torch.FloatTensor
        """

        assert isinstance(feat, torch.Tensor)

        max_t = spec_sub_conf.get("max_t", 20)
        num_t_sub = spec_sub_conf.get("num_t_sub", 3)
        rate = spec_sub_conf.get("rate", 0)

        if rate < random.uniform(0, 1):

            y = feat.clone().detach()
            max_frames = y.size(0)
            for _ in range(num_t_sub):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                # only substitute the earlier time chosen randomly for current time
                pos = random.randint(0, start)
                y[start:end, :] = feat[start - pos:end - pos, :]
            return y
        else:
            return feat

    def spec_trim(self, feat, spec_trim_conf):
        """
        funciton:
            Trim tailing frames. Inplace operation.
            ref: TrimTail [https://arxiv.org/abs/2211.00522]

        Args:
            feat:  torch.FloatTensor
            spec_trim_conf

        Returns:
            feat:  torch.FloatTensor
        """
        assert isinstance(feat, torch.Tensor)

        max_t = spec_trim_conf.get("max_t", 20)
        rate = spec_trim_conf.get("rate", 0.0)

        if rate < random.uniform(0, 1):
            max_frames = feat.size(0)
            length = random.randint(1, max_t)
            if length < max_frames / 2:
                feat = feat.clone().detach()[:max_frames - length]
        return feat

    def filter(self, filter_conf):

        """ Filter sample according to feature and label length
            Inplace operation.

            Args::
                sample: {key, wav, label, sample_rate, ...}]
                max_length: drop utterance which is greater than max_length(10ms)
                min_length: drop utterance which is less than min_length(10ms)
                token_max_length: drop utterance which is greater than
                    token_max_length, especially when use char unit for
                    english modeling
                token_min_length: drop utterance which is
                    less than token_max_length
                min_output_input_ratio: minimal ration of
                    token_length / feats_length(10ms)
                max_output_input_ratio: maximum ration of
                    token_length / feats_length(10ms)

            Returns:
                bool: True to keep, False to filter
        """

        max_length = filter_conf.get("max_length", '4000')
        # print("************max_length****************:{}".format(max_length))
        min_length = filter_conf.get('min_length', 10)
        token_max_length = filter_conf.get('token_max_length', 200)
        token_min_length = filter_conf.get('token_min_length', 1)
        min_output_input_ratio = filter_conf.get("min_output_input_ratio", 0.0005)
        max_output_input_ratio = filter_conf.get("max_output_input_ratio", 1)

        t_filelist = []
        for i, f in enumerate(self.files):
            if "start" in f and "end" in f:
                # sample['wav'] is torch.Tensor, we have 100 frames every second
                duration = (f["end"] - f["start"]) * 100  # 10 ms

            else:
                wav_info = torchaudio.info(f["wav"])
                duration = wav_info.num_frames / wav_info.sample_rate * 100
            # print("key:{} duration:{}".format(f["key"], duration))
            if duration < min_length:
                continue
            if duration > max_length:
                # print(f['wav'])
                continue

            txt = f["txt"]
            txt_len = len(txt)

            if txt_len < token_min_length:
                continue
            if txt_len > token_max_length:
                continue
            '''
            if txt_len / duration < min_output_input_ratio:
                t_filelist.remove(f)
                continue
            if  txt_len / duration > max_output_input_ratio:
                t_filelist.remove(f)
                continue
            '''
            f["duration"] = duration
            t_filelist.append(f)

        return t_filelist

    def sortD(self, reverse=False):

        '''
        reverse=False：升序排序（默认值）
        '''
        sorted_list = sorted(self.files, key=lambda x: x["duration"], reverse=reverse)

        return sorted_list

    def __getitem__(self, index):

        f = self.files[index]

        if "start" in f.keys():
            waveform, orig_sr = self.readwav(f["wav"], f["start"], f["end"])
        else:
            waveform, orig_sr = self.readwav(f["wav"])

        if orig_sr == self.conf.get("sample_rate", 16000):
            pass
        else:
            waveform = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.conf.get("sample_rate", 16000))(waveform)

        # add noise
        if self.conf["augment"]["add_noise"]:
            waveform = self.add_noise(waveform, self.conf["augment"]["add_noise_conf"], self.conf["sample_rate"])
            # logging.info("add_noise, case1 isinf:{}\t case2 isnan:{}".format(torch.sum(waveform.isinf()), torch.sum(waveform.isnan())))
            # logging.info("add_noise_after:{}".format(waveform.shape))
        # add reverb
        if self.conf["augment"]["add_reverb"]:
            waveform = self.add_reverb(waveform, self.conf["augment"]["add_reverb_conf"],
                                       self.conf.get("sample_rate", 16000))
            # logging.info("add_reverb, case1 isinf:{}\t case2 isnan:{}".format(torch.sum(waveform.isinf()), torch.sum(waveform.isnan())))
            # logging.info("add_reverb_after:{}".format(waveform.shape))

        if self.conf["augment"]["wav_distortion"]:
            # logging.info("input_waveform type:{}".format(type(waveform)))
            waveform = self.wav_distortion(waveform, self.conf["augment"]["wav_distortion_conf"],
                                           self.conf.get("sample_rate", 16000))
            # logging.info("wav_distortion, case1 isinf:{}\t case2 isnan:{}".format(torch.sum(waveform.isinf()),
            # torch.sum(waveform.isnan()))) logging.info("add_distortion_after:{}".format(waveform.shape))

        if self.conf["augment"].get("speed_perturb"):
            waveform = self.speed_perturb(waveform, self.conf.get("sample_rate", 16000))
            # logging.info("speed_perturb, case1 isinf:{}\t case2 isnan:{}".format(torch.sum(waveform.isinf()),
            # torch.sum(waveform.isnan())))
        feat = self.compute_feat(waveform).squeeze(0)
        # print("feat.shape:{} feat_data:{}".format(feat.shape, feat)) logging.info("compute_feat, case1 isinf:{}\t
        # case2 isnan:{}".format(torch.sum(feat.isinf()), torch.sum(feat.isnan())))

        # spec_augment
        if self.conf["augment"]["spec_aug"]:
            feat = self.spec_aug(feat, self.conf["augment"]["spec_aug_conf"])
            # logging.info("spec_aug, case1 isinf:{}\t case2 isnan:{}".format(torch.sum(feat.isinf()), torch.sum(
            # feat.isnan())))

        # spec_sub
        if self.conf["augment"]["spec_sub"]:
            feat = self.spec_sub(feat, self.conf["augment"]["spec_sub_conf"])
            # logging.info("spec_sub, case1 isinf:{}\t case2 isnan:{}".format(torch.sum(feat.isinf()), torch.sum(
            # feat.isnan()))) logging.info("spec_sub_after:{}".format(feat.shape))

        # spec_trim
        if self.conf["augment"]["spec_trim"]:
            feat = self.spec_trim(feat, self.conf["augment"]["spec_trim_conf"])
            # logging.info("spec_trim, case1 isinf:{}\t case2 isnan:{}".format(torch.sum(feat.isinf()), torch.sum(
            # feat.isnan()))) logging.info("spec_trim_after:{}".format(feat.shape))

        if self.tokenizer:
            label = self.tokenizer.tokens2ids(f["txt"])
            label = torch.tensor(label, dtype=torch.int32)
        else:
            label = torch.zeros(1)

        return f["key"], feat, label

    def __len__(self):
        return len(self.files)


def mycollate_fn(data):
    feats = []
    targets = []
    keys = []

    for (key, feat, target) in data:
        keys.append(key)
        feats.append(feat)
        targets.append(target)

    padded_x_lens = torch.tensor([feat.shape[0] for feat in feats], dtype=torch.int32)
    padded_y_lens = torch.tensor([targ.shape[0] for targ in targets], dtype=torch.int32)

    padded_x = pad_sequence(
        feats,
        batch_first=True,
        padding_value=0
    )

    padded_y = pad_sequence(
        targets,
        batch_first=True,
        padding_value=-1
    )

    return keys, padded_x, padded_x_lens, padded_y, padded_y_lens


def init_dataset_and_dataloader(args, tokenizer=None, seed=4233):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_conf = args["data_conf"]

    dev_conf = copy.deepcopy(train_conf)

    dev_conf['shuffle'] = False
    dev_conf["augment"]["speed_perturb"] = False
    dev_conf["augment"]["wav_distortion"] = False
    dev_conf["augment"]["add_reverb"] = False
    dev_conf["augment"]["add_noise"] = False
    dev_conf["augment"]['spec_aug'] = False
    dev_conf["augment"]['spec_sub'] = False
    dev_conf["augment"]['spec_trim'] = False
    dev_conf["filter"] = False
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    train_set = Dataload(args["train_file"], train_conf, tokenizer=tokenizer)
    dev_set = Dataload(args["dev_file"], dev_conf, tokenizer=tokenizer)

    '''
    
    shuffle=True
        随机性和重复性：
        可以选择是否在每个epoch内对数据进行重新排序或随机化。

    '''
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, shuffle=train_conf.get("shuffle"), rank=rank)

    '''
    prefetch_factor:
        参数指定了预取的批次数量。
        例如，如果设置为2，则DataLoader会提前加载两个批次的数据到内存中。
        这对于I/O密集型操作（如从磁盘读取数据）特别有效，因为它可以利用CPU和GPU并行处理的能力，从而减少数据加载造成的训练延迟.


    '''
    train_loader = DataLoader(train_set,
                              batch_size=train_conf.get("batch_size", 1),
                              pin_memory=train_conf.get("pin_memory", False),
                              num_workers=train_conf.get("num_workers", 0),
                              persistent_workers=True,
                              generator=generator,
                              sampler=train_sampler,
                              collate_fn=mycollate_fn,
                              # shuffle=train_conf.get("shuffle"),
                              prefetch_factor=train_conf.get("prefetch", 500)
                              )

    dev_loader = DataLoader(dev_set,
                            batch_size=dev_conf.get("batch_size", 1),
                            pin_memory=dev_conf.get("pin_memory", False),
                            num_workers=dev_conf.get("num_workers", 1),
                            persistent_workers=True,
                            generator=generator,
                            collate_fn=mycollate_fn,
                            shuffle=dev_conf.get("shuffle"),
                            prefetch_factor=args.get("prefetch", 500)
                            )

    return train_set, train_loader, train_sampler, dev_set, dev_loader

'''
configs = {
    "train_file": "data_copy/train/data.list",
    "dev_file": "data_copy/dev/data.list",
    "data_conf": {
        "sample_rate": 16000,
        "filter": True,
        "filter_conf": {
            "max_length": 2400,
            "min_length": 10,
            "token_max_length": 200,
            "token_min_length": 1,
            "min_output_input_ratio": 0.005,
            "max_output_input_ratio": 1
        },

        "feat_type": "fbank",
        "feat_conf": {
            "num_mel_bins": 80,
            "frame_shift": 10,
            "frame_length": 25,
            "dither": 0.1
        },
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "prefetch": 500,
        "shuffle": True,
        "augment": {
            "spec_aug": False,
            "spec_aug_conf": {
                "num_t_mask": 2,
                "num_f_mask": 2,
                "max_t": 50,
                "max_f": 10,
                "rate": 0.5,
            },
            "spec_sub": False,
            "spec_sub_conf": {
                "max_t": 20,
                "num_t_sub": 3,
                "rate": 0.5,
            },
            "spec_trim": False,
            "spec_trim_conf": {
                "max_t": 20,
                "rate": 0.5,
            },
            "speed_perturb": False,
            "add_noise": False,
            "add_noise_conf": {
                "noise_lists": "data/noise/musan.lst",
                "snr_db": [5, 10, 15],
                "rate": 0.5,
            },
            "add_reverb": False,
            "add_reverb_conf": {
                "reverb_lists": "data/noise/rirs.lst",
                "rate": 0.5,
            },
            "wav_distortion": False,
            "wav_distortion_conf": {
                "rate": 0.5,
                "gain_db": {
                    "db": -30,
                },
                "max_distortion": {
                    "max_db": -30,
                },
                "jag_distortion": {
                    "mask_number": 4,
                },
                "fence_distortion": {
                    "mask_number": 1,
                    "max_db": -30
                },
                "poly_distortion": {
                    "a": 4,
                    "m": 2,
                    "n": 2
                },
                "quad_distortion": {
                    "a": 1,
                    "m": 1,
                    "n": 1
                },
                "none_distortion": {

                }
            }

        }
    }
}

print(configs)
tokenizer = Tokenizers(configs.get("train_file"))
train_set, train_loader, train_sampler, dev_set, dev_loader = init_dataset_and_dataloader(configs, tokenizer=tokenizer,
                                                                                          seed=4233)
for i in range(10):
    print("********************{}********************".format(i))
    train_sampler.set_epoch(i)
    for batch_idx, batch_data in enumerate(train_loader):
        keys, padded_x, padded_x_lens, padded_y, padded_y_lens = batch_data
        print(padded_y, padded_y_lens)

        print("keys:{}, padded_x_lens:{}, padded_x.shape:{}".format(keys, padded_x_lens, padded_x.shape))
'''
