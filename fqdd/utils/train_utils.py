import logging
import os

import torch
import torch.nn as nn
import torch.distributed as dist


def init_distributed(dist_conf):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    logging.info('training on multiple gpus, this gpu {}'.format(local_rank) +
                 ', rank {}, world_size {}'.format(rank, world_size))
    if dist_conf.get("train_engine") in ["torch_ddp", "torch_fsdp"]:
        logging.info("############### local_rank:{} ###################".format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(dist_conf.get("dist_backend"))
    else:
        logging.error("not supported engine: {}".format(dist_conf.get("train_engine")))
    return world_size, local_rank, rank


def init_model(model_conf, init_method: str = ""):
    model = EBranchformer(model_conf)
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

    return model
