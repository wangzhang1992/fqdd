import os
import numpy as np
import json
import torch
import random
import traceback
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from .lang import read_phones
from .feature import extract_feat

'''
My_Data("result_test/2021/dev.json", {"1":2})

'''


class My_Data(Dataset):
    def __init__(
            self,
            json_path,
            phones,
            feat_type="mfcc",
            feat_cof=40,
            min_during=0.2,
            max_during=25,
            max_t=20,
            num_t_sub=3,
            min_trans_len=0,
            max_trans_len=128
    ):

        self.json_path = json_path
        self.phones = phones
        self.feat_type = feat_type
        self.feat_cof = feat_cof
        self.min_during = min_during
        self.max_during = max_during
        self.min_trans_len = min_trans_len
        self.max_trans_len = max_trans_len
        self.max_t = max_t
        self.num_t_sub = num_t_sub
        self.json_data = {}

        try:
            # {1:{path:"data/train/1.wav", trans:"上 热 搜"}}
            jsons = json.load(open(self.json_path, 'r', encoding='utf-8'))
            num = 0
            for item in jsons:
                '''
               if "train" in self.json_path and len(self.json_data)>50:
                   continue
               if "dev" in self.json_path and len(self.json_data)>10:
                   continue
               '''
                during = jsons[item]["during"]
                trans_len = len(jsons[item]["trans"].replace(' ', ''))
                if self.min_during < during and during < self.max_during and self.min_trans_len < trans_len and self.max_trans_len > trans_len:
                    if num < 12100:
                        self.json_data[len(self.json_data)] = jsons[item]
                        num += 1
        except:
            print("json_path None:{}".format(self.json_path))

    def __getitem__(self, index):

        target = [self.phones[w] for w in self.json_data[index]["trans"]]
        feat = extract_feat(self.json_data[index]["path"], feat_type=self.feat_type, feat_cof=self.feat_cof)
        # print(feat.shape)
        if "train" in self.json_path:
            feat = self.spec_sub(feat, max_t=self.max_t, num_t_sub=self.num_t_sub)
            feat = self.spec_strim(feat, max_t=self.max_t)
        return feat.numpy(), target

    def __len__(self):
        return len(self.json_data)

    def spec_strim(self, feat, max_t=20):
        assert isinstance(feat, torch.Tensor)
        max_frames = feat.size(0)
        length = random.randint(1, max_t)
        if length < max_frames / 2:
            feat = feat.clone().detach()[:max_frames - length]
        return feat

    def spec_sub(self, feat, max_t=20, num_t_sub=3):
        assert isinstance(feat, torch.Tensor)
        y = feat.clone().detach()
        max_frames = y.size(0)
        for _ in range(num_t_sub):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            pos = random.randint(0, start)
            y[start:end, :] = feat[start - pos:end - pos, :]
        return y


class Load_Data:
    def __init__(
            self,
            phones,
            params,
    ):
        self.phones = phones
        self.params = params

        self.respath = os.path.join(self.params.result_dir, str(self.params.seed))
        self.train_json_path = os.path.join(self.respath, "train.json")
        self.test_json_path = os.path.join(self.respath, "test.json")
        self.dev_json_path = os.path.join(self.respath, "dev.json")

        self.train_load = None
        self.test_load = None
        self.dev_load = None
        self.train_sampler = None

    def dataload(self):
        if os.path.exists(self.train_json_path):
            train_data = My_Data(self.train_json_path, self.phones, feat_type=self.params.feat_type,
                                 feat_cof=self.params.feat_cof)
            if self.params.is_distributed:
                self.train_sampler = DistributedSampler(train_data)
                self.train_load = DataLoader(dataset=train_data, sampler=self.train_sampler,
                                             batch_size=self.params.batch_size, pin_memory=True,
                                             num_workers=self.params.num_workers, collate_fn=self.collate_fn,
                                             drop_last=True)
            else:
                self.train_load = DataLoader(dataset=train_data, batch_size=self.params.batch_size,
                                             shuffle=self.params.shuffle, pin_memory=True,
                                             num_workers=self.params.num_workers, collate_fn=self.collate_fn,
                                             drop_last=True)

        if os.path.exists(self.test_json_path):
            test_data = My_Data(self.test_json_path, self.phones, feat_type=self.params.feat_type,
                                feat_cof=self.params.feat_cof)
            self.test_load = DataLoader(dataset=test_data, batch_size=self.params.batch_size,
                                        num_workers=self.params.num_workers, shuffle=False, collate_fn=self.collate_fn,
                                        drop_last=False)
        if os.path.exists(self.dev_json_path):
            dev_data = My_Data(self.dev_json_path, self.phones, feat_type=self.params.feat_type,
                               feat_cof=self.params.feat_cof)
            self.dev_load = DataLoader(dataset=dev_data, batch_size=self.params.batch_size,
                                       num_workers=self.params.num_workers, shuffle=False, collate_fn=self.collate_fn)

    def collate_fn(self, data):
        feats = []
        targets = []
        targets_sos = []
        targets_eos = []
        wav_lengths = []
        target_lens = []
        target_os_lens = []

        for (feat, target) in data:
            wav_length = feat.shape[0]
            target_len = len(target)
            target_sos = [self.phones["<sos/eos>"]] + target
            target_eos = target + [self.phones["<sos/eos>"]]

            feats.append(feat)
            targets.append(target)
            wav_lengths.append(wav_length)
            target_lens.append(target_len)
            targets_sos.append(target_sos)
            targets_eos.append(target_eos)
            target_os_lens.append(target_len + 1)
        # print("##################step1#######################")       
        feats = self.zero_pad_concat(feats)
        # print(feats.shape)
        targets = self.end_pad_concat(targets)
        targets_sos = self.end_pad_concat(targets_sos)
        targets_eos = self.end_pad_concat(targets_eos)

        wav_lengths = self.norm_len(wav_lengths)
        target_lens = self.norm_len(target_lens)
        target_os_lens = self.norm_len(target_os_lens)
        # print("#######################step2########################")
        # print(feats.shape)
        feats = self.norm_data(feats)
        # print("########################after norm_data##########################")
        # print(feats.shape)
        return feats, targets, targets_sos, targets_eos, wav_lengths, target_lens, target_os_lens

    def zero_pad_concat(self, inputs):
        # print("###############zero")
        # feature_type == "raw" 
        if len(inputs[0].shape) == 1:
            shape = (len(inputs), max(item.shape[0] for item in inputs))
        else:
            # feature_type = "mfcc/fbank"
            shape = (len(inputs), max(item.shape[0] for item in inputs), inputs[0].shape[-1])

        input_mats = []
        input_mat = np.zeros(shape, dtype=np.float32)
        for e, inp in enumerate(inputs):
            input_mat[e, :len(inp)] = inp
            # for index, l in enumerate(inp):
            #    input_mat[e, index] = l
        return torch.FloatTensor(input_mat)

    '''
    inputs = [np.random.randn(2,5), np.random.randn(3,5), np.random.randn(6,5), np.random.randn(5,5)]
    print(inputs)
    print(zero_pad_concat(inputs))
    '''

    def end_pad_concat(self, inputs):
        shape = (len(inputs), max([len(item) for item in inputs]))
        target = np.full(shape, fill_value=0, dtype='i')
        for e, l in enumerate(inputs):
            for index, item in enumerate(l):
                target[e, index] = item
        return torch.LongTensor(target)

    '''
    inputs = [[2,4,1,2],[3,1,2],[1,5,7,2,7,7],[1,2]]
    print(inputs)
    print(end_pad_concat(inputs))
    '''

    def norm_data(self, x, dim=-1):
        try:
            x = (x - x.mean(dim, keepdim=True)) / (1e-10 + x.std(dim, keepdim=True))
        except:
            traceback.print_exc()
        return x

    def norm_len(self, inputs):
        inputs = torch.FloatTensor(inputs)
        return torch.FloatTensor(inputs / torch.max(inputs))


def load_txt_data(label_path, args, shuffle=False):
    if args.label_status == 'word':
        phones = read_phones(args.phones_path)
        # my_data = Data_Text_Set(label_path, phones=phones)

    elif args.label_status == "phones":
        w2p = read_phones(args.lexicon)
        # my_data = Data_Text_Set(label_path, lexicon=args.lexicon, w2p=w2p)

    else:
        raise Exception("set correct label_Status")

    # return DataLoader(dataset=my_data, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,
    #                   drop_last=True)


def test():
    dic = {}
    if os.path.exists('/data/work/own_learn/result/2021/phones.txt'):
        with open('/data/work/own_learn/result/2021/phones.txt', 'r', encoding='utf-8') as rf:

            for line in rf:
                lines = line.strip().split('\t')
                dic[lines[0]] = int(lines[1])

    test_data = My_Data('/data/work/own_learn/result/2021/dev.json', dic)
    dev_data = DataLoader(dataset=test_data, batch_size=1, pin_memory=True, num_workers=0, drop_last=False)
    for idx, data in enumerate(tqdm(dev_data)):
        print(idx)
        print(data)
        feats, targets = data
        print("feat.shape:{}, label:{}".format(feats.shape, targets))

# test()
