import torch.nn as nn

from fqdd.models.conformer.conformer import Conformer
from fqdd.module_nnets.model_utils import load_checkpoint
from fqdd.models.ebranchformer.ebranchformer import EBranchformer
from fqdd.models.crdnn.CRDNN import Encoder_Decoer
from fqdd.models.las.las import LAS

MODEL_LISTS = {
    "conformer": Conformer,
    "crdnn": Encoder_Decoer,
    "ebranchformer": EBranchformer,
    "las": LAS
}


def model_params_init(model, init_method="default"):
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


def init_model(args, configs):
    model_name = configs["model_name"]
    model_configs = configs["model"]
    model = MODEL_LISTS[model_name](model_configs)
    if args.checkpoint is not None:

        infos = load_checkpoint(model, args.checkpoint)
    else:
        init_method = model_configs.get("init_method", "default")
        model_params_init(model, init_method)
        infos = {}
    configs["init_infos"] = infos
    return model, configs
