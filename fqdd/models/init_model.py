import torch.nn as nn

from fqdd.models.conformer.conformer import Conformer
from fqdd.models.crdnn.crdnn import CRDNN
from fqdd.models.ebranchformer_ehance.ebranchformer import EBranchformer_Ehance
from fqdd.modules.model_utils import load_checkpoint
from fqdd.models.ebranchformer.ebranchformer import EBranchformer

MODEL_LISTS = {
    "conformer": Conformer,
    "ebranchformer": EBranchformer,
    "crdnn": CRDNN,
    "ebranchformer_ehance": EBranchformer_Ehance
}


def model_params_init(model, init_method="default"):
    if init_method == "kaiming":
        # He初始化方法是是Xavier初始化的一种变体, 针对ReLU激活函数的情况进行了优化。对于具有n个输入的层，参数可以从均匀分布或高斯分布中采样，并将方差设置为2 /
        # n。这种方法在使用ReLU等激活函数时表现良好。通过控制权重的方差来避免梯度消失或梯度爆炸问题。
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    elif init_method == "xavier_uniform":
        # 均匀分布的区间为[-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]
        # Xavier初始化方法根据每一层的输入和输出的维度来确定参数的初始值。对于具有n个输入和m个输出的层，参数可以从均匀分布或高斯分布中采样，并将方差设置为2 / (n +
        # m)，适用于sigmoid和tanh的网络。它通过控制权重的方差来避免梯度消失或梯度爆炸问题。对于非ReLU激活函数的网络，效果可能不佳。
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    elif init_method == "xavier_normal":
        # 正态分布的均值为0、方差为sqrt( 2/(fan_in + fan_out) )
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)

    elif init_method == "Orthogonal":
        # 正交初始化（Orthogonal Initialization）
        # 解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用的参数初始化方法
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(m.weight)

    elif init_method == "default":
        # 在PyTorch中，默认的初始化方式是均匀分布（Uniform Distribution）进行初始化。具体来说，权重参数的初始值在 [-sqrt(k), sqrt(k)] 之间均匀分布，其中 k = 1 /
        # in_features。
        pass
    else:
        assert "model init method error"


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
