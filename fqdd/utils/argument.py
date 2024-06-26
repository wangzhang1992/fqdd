import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'torch_fsdp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser = add_model_args(parser)
    parser = add_dataset_args(parser)
    parser = add_ddp_args(parser)
    parser = add_lora_args(parser)
    parser = add_deepspeed_args(parser)
    parser = add_fsdp_args(parser)
    parser = add_trace_args(parser)
    args = parser.parse_args()
    if args.train_engine == "deepspeed":
        args.deepspeed = True
        assert args.deepspeed_config is not None
    return args


def add_model_args(parser):
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument("--enc_init",
                        default=None,
                        type=str,
                        help="Pre-trained model to initialize encoder")
    parser.add_argument(
        '--enc_init_mods',
        default="encoder.",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of encoder modules \
                        to initialize ,separated by a comma")
    parser.add_argument(
        '--freeze_modules',
        default="",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help='free module names',
    )
    return parser


def add_trace_args(parser):
    parser.add_argument('--jit',
                        action='store_true',
                        default=False,
                        help='if use jit to trace model while training stage')
    parser.add_argument('--print_model',
                        action='store_true',
                        default=False,
                        help='print model')
    return parser


def add_dataset_args(parser):
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    return parser


def add_lora_args(parser):
    parser.add_argument("--use_lora",
                        default=False,
                        type=bool,
                        help="whether use the lora finetune.")
    parser.add_argument("--only_optimize_lora",
                        default=False,
                        type=bool,
                        help="freeze all other paramters and only optimize \
                        LoRA-related prameters.")
    parser.add_argument("--lora_list",
                        default=['o', 'q', 'k', 'v'],
                        help="lora module list.")
    parser.add_argument("--lora_rank",
                        default=8,
                        type=int,
                        help="lora rank num.")
    parser.add_argument("--lora_alpha",
                        default=8,
                        type=int,
                        help="lora scale param, scale=lora_alpha/lora_rank.")
    parser.add_argument("--lora_dropout",
                        default=0,
                        type=float,
                        help="lora dropout param.")
    return parser


def add_ddp_args(parser):
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--fp16_grad_sync',
                        action='store_true',
                        default=False,
                        help='Use fp16 gradient sync for ddp')
    return parser


def add_deepspeed_args(parser):
    parser.add_argument('--timeout',
                        default=30,
                        type=int,
                        help='timeout (in seconds) of wenet_join. ' +
                             '30s for aishell & 300s for wenetspeech')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    # DeepSpeed automaticly add '--deepspeed' and '--deepspeed_config' to parser
    parser = deepspeed.add_config_arguments(parser)
    return parser


def add_fsdp_args(parser):
    parser.add_argument(
        '--dtype',
        default='fp32',
        choices=['fp32', 'fp16', 'bf16'],
        help='when amp is used, dtype is automatically set to fp16.\
        this arg has no effect when deepspeed is enabled.')
    parser.add_argument(
        '--fsdp_cpu_offload',
        default=False,
        type=bool,
        help='whether to offload parameters to CPU',
    )
    parser.add_argument(
        '--fsdp_sync_module_states',
        type=bool,
        default=True,
        help='\
        each FSDP module will broadcast module parameters and buffers from \
        rank 0 to ensure that they are replicated across ranks',
    )
    parser.add_argument(
        '--fsdp_sharding_strategy',
        default='zero2',
        # TODO(Mddct): pipeline and model parallel (3-D parallelism)
        choices=['no_shard', 'model', 'zero2', 'zero3'],
        help='Sharding strategy for FSDP. Choose from the following options:\n'
             '  - "no_shard": Equivalent to DistributedDataParallel (DDP).\n'
             '  - "model": WENET_ENC_DEC strategy, equivalent to DeepSpeed zero1.\n'
             '  - "zero2": SHARD_GRAD_OP strategy, equivalent to DeepSpeed zero2.\n'
             '  - "zero3": FULL_SHARD strategy, equivalent to DeepSpeed zero3.\n'
             'For more information, refer to the FSDP API documentation.')
    return parser
