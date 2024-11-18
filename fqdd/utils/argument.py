import argparse
import os


def parse_arguments():
    model_args = argparse.ArgumentParser(
        description='model config', add_help=True)
    model_args.add_argument(
        '--seed',
        default=2024,
        type=int,
        help='rand seed'
    )

    model_args.add_argument(
        '--train_config',
        type=str,
        default="conf/ebranchformer_conf.json",
        help="模型配文件"
    )

    model_args.add_argument(
        '--is-distributed',
        type=bool,
        default=True,
        help="是否分布式训练"
    )

    model_args.add_argument(
        '--world-size',
        type=int,
        default=3,
        help='设置训练GPU卡数'

    )

    model_args.add_argument(
        '--local-rank',
        type=int,
        default=0,
        help="进程编号"
    )

    model_args.add_argument(
        '--host',
        type=str,
        default="env://localhost:2345",
        # default="env://",
        help="设置主机服务器ip"
    )

    model_args.add_argument(
        '--tensorboard_dir',
        default='log/tensorboard',
        type=str,
        help='tensorbard show dir'
    )

    model_args.add_argument(
        '--data_folder',
        # default="test_folder",
        default="/data1/data_management/speech_processing/ASR/audio_raw/language_china/zh-CN/lable/near-field/read/aishell/aishell_1_178hr/data_aishell/",
        type=str,
        help='data_folder path'
    )

    model_args.add_argument('--train_data', required=True, help='train data file')
    model_args.add_argument('--dev_data', required=True, help='cv data file')
    model_args.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='model save path'
    )
    model_args.add_argument(
        '--checkpoint',
        type=str,
        help='预加载模型路径'
    )

    model_args.add_argument(
        '--max_epoch',
        default=120,
        type=int,
        help='train epoch'
    )

    model_args.add_argument('--train_engine',
                            default='torch_ddp',
                            choices=['torch_ddp', 'torch_fsdp', 'deepspeed'],
                            help='Engine for paralleled training'
                            )

    model_args.add_argument('--ddp.dist_backend',
                            dest='dist_backend',
                            default='nccl',
                            choices=['nccl', 'gloo', "hccl"],
                            help='distributed backend'
                            )

    model_args.add_argument('--device',
                            type=str,
                            default='cuda',
                            choices=["cpu", "npu", "cuda"],
                            help='accelerator for training'
                            )

    model_args.add_argument(
        '--batch_size',
        default=1,
        type=int,
        help='batch set'
    )

    model_args.add_argument(
        '--num_workers',
        default=8,
        type=int,
        help='set thread number reading data '
    )

    model_args.add_argument(
        '--use_lm',
        default=False,
        type=bool,
        help='use lm'
    )

    model_args.add_argument(
        '--opt_level',
        default="O1",
        type=str,
        help='训练数据精度, '
             'O0 : 执行FP32训练'
             'O1 : 当前使用部分FP16混合训练'
             'O2 : 除了BN层的权重外，其他层的权重都使用FP16执行训练'
             'O3 : 默认所有的层都使用FP16执行计算，当keep_batch norm_fp32=True，'
             '则会使用cudnn执行BN层的计算，该优化等级能够获得最快的速度，但是精度可能会有一些较大的损失。'

    )

    return model_args.parse_args()


def reload_configs(args, configs):
    configs["model_dir"] = args.model_dir
    configs["max_epoch"] = args.max_epoch if "max_epoch" not in str(configs) else configs["max_epoch"]
    configs["data_conf"]["batch_size"] = args.batch_size if "batch_size" not in str(configs) else configs["data_conf"][
        "batch_size"]
    # print(configs)
    return configs
