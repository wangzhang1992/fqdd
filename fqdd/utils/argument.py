import argparse

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
        '--config',
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
        #default="env://",
        help="设置主机服务器ip"
    )

    model_args.add_argument(
        '--tensorboard_dir',
        default='log/tensorboard',
        type=str,
        help= 'tensorbard show dir'
    )

    model_args.add_argument(
        '--data_folder',
        # default="test_folder",
        default="/data1/data_management/speech_processing/ASR/audio_raw/language_china/zh-CN/lable/near-field/read/aishell/aishell_1_178hr/data_aishell/", 
        type=str,
        help='data_folder path'
    )

    model_args.add_argument(
        '--max_epoch',
        default=200,
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
        default=8,
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
        '--lr',
        default=0.001,
        type=float,
        help='set learning rate '
    )

    model_args.add_argument(
        '--lexicon',
        default='dic/lexicon.txt',
        type=str,
        help='set lexicon path'
    )

    model_args.add_argument(
        '--label_status',
        default='word',
        type=str,
        help=' label status: word/phones '
    )

    model_args.add_argument(
        '--min_trans_len',
        default=1,
        type=int,
        help='max_label_length'
    )

    model_args.add_argument(
        '--max_trans_len',
        default=50,
        type=int,
        help='max_label_length'
    )

    model_args.add_argument(
        '--e_num_layers',
        default=4,
        type=int,
        help='net layer size'
    )


    model_args.add_argument(
        '--result_dir',
        default='exp',
        type=str,
        help='result path'
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

    model_args.add_argument(
        '--amp',
        default=False,
        type=bool,
        help='use apm'
    )

    model_args.add_argument(
        '--embedding_dim',
        default=512,
        type=int,
        help='Embedding nodes'
    )

    model_args.add_argument(
        '--pretrained',
        default=True,
        type=bool,
        help="是否加载预训练模型"
    )

    model_args.add_argument(
        '--lm_path',
        default="lm/lm_best.pth",
        type=str,
        help='lm path'
    )



    return model_args.parse_args()
