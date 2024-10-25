import torch
import torch.nn as nn
import numpy as np
from script.tools.argument import parse_arguments
from script.asr.decode import int2word
from script.tools.lang import read_phones
from script.lm.lm_net import RNN_Model
from script.tools.load_data import end_pad_concat


def load_model(args, dic, device):
    lm_model = RNN_Model(len(dic), args)
    try:
        lm_model.to(device)
        checkpoint = torch.load(args.reload_lm_model)
        lm_model.load_state_dict(checkpoint['model'])
    except:
        print("set correct model_path")
    finally:
        return lm_model.eval()


def inference(lm_model, phones, strs, device):
    label_int = np.array([[phones[item] for item in strs if item in phones.keys()]])
    pad_label = end_pad_concat(label_int, max_length=128)
    test_data = torch.LongTensor(pad_label).to(device)
    print(test_data)
    pre = lm_model(test_data)
    res = torch.argmax(pre, dim=2)
    return [item for item in int2word(res[0], phones) if item != '<ese>']


def main():
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    args = parse_arguments()
    dic = read_phones(args.phones_path)
    lm_model = load_model(args, dic, device)

    strs = '初始 学习率过小 导致 收脸慢，应曾大学习率，兵从头 开始训练初始 学习率郭小 导致 收敛慢，应增大学习率，并从头 开始训练'
    res = inference(lm_model, dic, strs, device)
    print(res)


if __name__ == '__main__':
    main()
