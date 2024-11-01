import json
import re

import torch
import os, sys
import torch.nn as nn
import time

'''
init model
'''


def model_init(model, init_method="default"):
    # 在ReLU网络中，假定每一层有一半的神经元被激活，另一半为0。推荐在ReLU网络中使用
    if init_method == "kaiming":
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    # 通用方法，适用于任何激活函数
    elif init_method == "default":
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    # 正交初始化（Orthogonal Initialization）
    # 解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用的参数初始化方法
    elif init_method == "Orthogonal":
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(m.weight)
    else:
        print("model init method error")
        sys.exit()


'''
save model
'''


def save_model(model_dir, model, checkpoint: int = -1, steps=-1):
    # assert model == torch.nn.Module

    os.makedirs(model_dir, exist_ok=True)

    if checkpoint == -1:
        checkpoint_name = "init"
    else:
        checkpoint_name = checkpoint

    model_path = os.path.join(model_dir, "model_" + str(checkpoint_name) + ".pt")
    config_path = os.path.join(model_dir, "model_" + str(checkpoint_name) + ".json")

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    conf_dict = {
        "checkpoint": checkpoint,
        "steps": steps
    }
    torch.save(state_dict, model_path)

    with open(config_path, "w") as wf:
        json.dump(conf_dict, wf, indent=4)


def load_model(model, model_dir, pretrain_model):
    """
    reload model
    """
    model_path = model_dir + "/" + pretrain_model
    config_path = re.sub('.pt$', '.json', model_path)

    model_dict = torch.load(model_path, map_location='cpu', mmap=True)
    conf_dict = json.load(open(config_path))
    model.load_state_dict(model_dict)

    checkpoint = conf_dict["checkpoint"] + int("init_" not in config_path)

    return model, checkpoint, conf_dict["steps"]


def infer_model(load_dir, model):
    # load epoch
    try:
        model_path = os.path.join(load_dir, 'model.ckpt')
        check_model = torch.load(model_path)
        model.load_state_dict(check_model['model'])
    except:
        print('reload_model, file not exists:\t{}'.format(epoch_path))
