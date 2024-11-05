import os
import logging
import copy
import deepseeed
import torch
import torch.distributed as dist
import torch.optim as optim
from fqdd.utils.optimizers import adam_optimizer, sgd_optimizer, scheduler, WarmupLR, NoamHoldAnnealing


def init_distributed(args):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    logging.info('training on multiple gpus, this gpu {}'.format(local_rank) +
                 ', rank {}, world_size {}'.format(rank, world_size))
    if args.train_engine in ["torch_ddp", "torch_fsdp"]:
        if "cuda" in args.device:
            torch.cuda.set_device(local_rank)
        elif "npu" in args.device and TORCH_NPU_AVAILABLE:
            torch.npu.set_device(local_rank)
        else:
            logging.error("not supported device: {}".format(args.device))
        dist.init_process_group(args.dist_backend)
    elif args.train_engine == "deepspeed":
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    else:
        logging.error("not supported engine: {}".format(args.train_engine))
    return world_size, local_rank, rank


def init_optimizer_and_scheduler(configs, model):
    params = model.parameters()
    optim_conf = copy.deepcopy(configs['optim_conf'])
    if configs['optim'] == 'adam':
        optimizer = optim.Adam(params, **optim_conf)
    elif configs['optim'] == 'adamw':
        optimizer = optim.AdamW(params, **optim_conf)
    else:
        raise ValueError("unknown optimizer: " + configs['optim'])

    scheduler_type = None
    if configs['scheduler'] == 'warmuplr':
        scheduler_type = WarmupLR
        scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    elif configs['scheduler'] == 'NoamHoldAnnealing':
        scheduler_type = NoamHoldAnnealing
        scheduler = NoamHoldAnnealing(optimizer, **configs['scheduler_conf'])
    else:
        raise ValueError("unknown scheduler: " + configs['scheduler'])

    step = configs["init_infos"].get("step", -1)
    scheduler.set_step(step)
    return model, optimizer, scheduler
