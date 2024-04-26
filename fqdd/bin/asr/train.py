import torch
import torch.nn as nn
import os, sys
import numpy as np
sys.path.insert(0, "./")
import torch.optim.lr_scheduler as lr_sch

from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
# from apex import amp
from tqdm import tqdm
from fqdd.utils.feature import get_feats
# from fqdd.prepare_data.aidatatang_prepare_data import prepare_data
from fqdd.prepare_data.aishell_prepare_data import prepare_data
# from fqdd.prepare_data.thch30_prepare_data import prepare_data
from fqdd.utils.lang import create_phones, read_phones
from fqdd.utils.load_data import Load_Data
from fqdd.utils.argument import parse_arguments
from fqdd.asr.decode import GreedyDecoder, calculate_cer
# from fqdd.models.wav2vec import Encoder_Decoer
from fqdd.models.CRDNN import Encoder_Decoer
from fqdd.models.check_model import model_init, save_model, reload_model
from fqdd.utils.optimizers import adam_optimizer, sgd_optimizer, scheduler, warmup_lr
from fqdd.utils.logger import init_logging
from fqdd.nnets.losses import nll_loss, transducer_loss
from fqdd.models.dense import densenet169
from fqdd.utils.edit_distance import Static_Info


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

def calculate_loss(ctc_loss, pred, gold, input_lengths, target_lengths):

    # print("{}\t{}\t{}\t{}".format(pred.shape, gold.shape, input_lengths.shape, target_lengths.shape))
    """
    Calculate loss
    args:
        pred: B x T x C
        gold: B x T
        input_lengths: B (for CTC)
        target_lengths: B (for CTC)
        smoothing:
        type: ce|ctc (ctc => pytorch 1.0.0 or later)
        input_lengths: B (only for ctc)
        target_lengths: B (only for ctc)
    """
    input_lengths = (input_lengths * pred.shape[1]).int()
    target_lengths = (target_lengths * gold.shape[1]).int()
    log_probs = pred.transpose(0, 1)  # T x B x C
    # print(gold.size())
    targets = gold
    # targets = gold.contiguous().view(-1)  # (B*T)

    """
    log_probs: torch.Size([209, 8, 3793])
    targets: torch.Size([8, 46])
    input_lengths: torch.Size([8])
    target_lengths: torch.Size([8])
    """
    
    # log_probs = F.log_softmax(log_probs, dim=2)
    # log_probs = log_probs.detach().requires_grad_()
    loss = ctc_loss(log_probs.to("cpu"), targets.to("cpu"), input_lengths.to("cpu"), target_lengths.to("cpu"))

    return loss


def train(model, load_object, args, phones, logger):

    min_loss = 1000
    lr_cer = 100
    slice_len = 5    
    epoch_n = args.epoch_n
    device = model.device
    train_sampler = load_object.train_sampler
    train_load = load_object.train_load
    dev_load = load_object.dev_load

    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean') 
    optimizer = adam_optimizer(model, args.lr)
    #optimizer = sgd_optimizer(model, args.lr)
    warm_up = warmup_lr(args.lr, 20000)
    scheduler_lr = scheduler(optimizer, patience=0, cooldown=0)
    
    if args.pretrained:
        # if args.local_rank < 1:
        start_epoch = reload_model(os.path.join(args.result_dir, str(args.seed), "save", "AM"), model=model, optimizer=optimizer, map_location='cuda:{}'.format(args.local_rank))
        #else:
            #start_epoch = reload_model(os.path.join(args.result_dir, str(args.seed), "save", "AM"))
        start_epoch = start_epoch + 1
    else:
        start_epoch = 1
 
    if args.local_rank < 1: 
        logger.info("init_lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
    accum_grad = 4
    for epoch in range(start_epoch, epoch_n+1):
        model.train()
        if isinstance(model, DDP):
            model_context = model.join
        else:
            model_context = nullcontext
        
        if args.is_distributed:
            train_sampler.set_epoch(epoch)
        if args.local_rank < 1:
            logger.info("Epoch {}/{}".format(epoch, epoch_n))
            logger.info("-" * 10)
        static = Static_Info()
        inter_cers = 0.0
        inter_num = 100
        if args.local_rank < 1:
            print("status: train\t train_load_size:{}".format(len(train_load)))
        dist.barrier() # 同步训练进程
        with model_context():
            for idx, data in enumerate(tqdm(train_load)):
                context = None
                if (idx+1) % accum_grad != 0:
                    # Used for single gpu training and DDP gradient synchronization processes.
                    context = model.no_sync
                else:
                    context = nullcontext

                with context():
                    # 只做推理，代码不会更新模型状态 
                    feats, targets, targets_bos, targets_eos, wav_lengths, target_lens, target_os_lens = [item.to(device) for item in data]
                    output_en, output_de = model(feats, targets_bos) 
                    output_en = torch.nn.functional.log_softmax(output_en, dim=len(output_en.shape)-1)
                    output_de = torch.nn.functional.log_softmax(output_de, dim=len(output_de.shape)-1)
                    closs = calculate_loss(ctc_loss, output_en, targets, wav_lengths, target_lens)
                    ce_loss = nll_loss(output_de, targets_eos, target_os_lens)
                    loss = closs*0.7 +ce_loss*0.3
                    # 看loss是不是nan,如果loss是nan,
                    # 那么说明可能是在forward的过程中出现了第一条列举的除0或者log0的操作
                    # print(loss)
                    loss.backward()
                    targ, pred = GreedyDecoder(output_en, targets, wav_lengths, target_lens, phones)
                    cer = static.one_iter(targ, pred, loss)
                    inter_cers += cer
                if (idx+1) % accum_grad == 0: 
                    # loss不是nan,那么说明forward过程没问题，可能是梯度爆炸，所以用梯度裁剪试试
                    torch.nn.utils.clip_grad_norm_((p for p in model.parameters()), max_norm=50)
                    optimizer.step() 
                optimizer.zero_grad()
                warm_up(optimizer)

                if (idx+1) % inter_num == 0 and (idx+1) % accum_grad == 0:
                    avg_inter_loss = static.get_inter_loss(inter_num)
                    avg_inter_cer = inter_cers / inter_num
                    inter_cers = 0.0
                    logger.info("batchs:{}   Loss:{:.2f}   CER:{:.2f}".format(idx+1, avg_inter_loss, avg_inter_cer))
                torch.cuda.empty_cache() 
            cer, corr, det, ins, sub = static.avg_one_epoch_cer()
            avg_one_epoch_loss = static.avg_one_epoch_loss(len(train_load))
    
            #scheduler_lr.step(loss)
            if args.local_rank < 1:
                logger.info("Epoch:{}, loss:{}, cer:{}, lr:{}, corr:{}, det:{}, ins:{}, sub:{}".format(epoch, avg_one_epoch_loss, cer, optimizer.state_dict()['param_groups'][0]['lr'], corr, det, ins, sub))
                save_model(model, optimizer, epoch, os.path.join(args.result_dir, str(args.seed), 'save', 'AM'))
            dist.barrier() # 同步测试进程
            with torch.no_grad():
                loss, cer, corr, det, ins, sub = evaluate(model, dev_load, ctc_loss, args, phones, device)
    
                #scheduler_lr.step(loss)
                #save_model(model, optimizer, epoch, os.path.join(args.result_dir, str(args.seed), 'save'))
                logger.info("DEV:  loss:{}, cer:{}, corr:{}, det:{}, ins:{}, sub:{}".format(loss, cer, corr, det, ins, sub))
            
    
def evaluate(model, eval_load, ctc_loss, args, phones, device):
    
    model.eval()
    dev_cer = Static_Info()

    for idx, data in enumerate(tqdm(eval_load)):
        feats, targets, targets_bos, targets_eos, wav_lengths, target_lens, target_os_lens = [item.to(device) for item in data]
        # print(feats.shape)
        output_en, output_de = model(feats, targets_bos)
        output_en = torch.nn.functional.log_softmax(output_en, dim=len(output_en.shape)-1)
        output_de = torch.nn.functional.log_softmax(output_de, dim=len(output_de.shape)-1)
        closs = calculate_loss(ctc_loss, output_en, targets, wav_lengths, target_lens)
        ce_loss = nll_loss(output_de, targets_eos, target_os_lens)
        loss = closs*0.7 +ce_loss*0.3
        torch.distributed.barrier()
        targ, pred = GreedyDecoder(output_en, targets, wav_lengths, target_lens, phones) 
        cer = dev_cer.one_iter(targ, pred, loss)
        torch.cuda.empty_cache()
    avg_dev_loss = dev_cer.avg_one_epoch_loss(len(eval_load))
    cer, corr, det, ins, sub = dev_cer.avg_one_epoch_cer()
    return avg_dev_loss, cer, corr, det, ins, sub

def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    
    # prepare data
    dirpath = os.path.join(args.result_dir, str(args.seed))
    prepare_data(args.data_folder, dirpath)
    phones = create_phones(dirpath)

    logger = init_logging("train", dirpath)

    if args.feat_type == 'mfcc':
        input_dim = args.feat_cof * 3
    elif args.feat_type == 'fbank':
        input_dim = args.feat_cof
    else:
        input_dim = -1

    output_dim = len(phones)

    #model = densenet169(num_classes=output_dim) 
    # model = Encoder_Decoer(output_dim, feat_shape=[args.batch_size, args.max_during*100, input_dim], output_size=1024)
    model = Encoder_Decoer(output_dim, feat_shape=[args.batch_size, args.max_during*100, input_dim])
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        if args.is_distributed:
            torch.cuda.set_device(args.local_rank)
            num_gpus = torch.cuda.device_count()
            # world_size #表示开启的全局进程个数进程数，node * num_gpus_one_node， 
            # rank 表示为第几个节点进程，一般rank=0表示，master节点
            # local_rank: 进程内，GPU 编号，非显式参数，由 torch.distributed.launch 内部指定。比方说， rank = 3，local_rank = 0 表示第 3 个进程内的第 1 块 GPU
            # dist.init_process_group(backend='nccl', init_method=args.host, rank=args.local_rank, world_size=args.world_size)
            dist.init_process_group(backend='nccl')
            device = torch.device('cuda', args.local_rank)
            model.to(device)
            print(num_gpus, device)
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    else:
        model = model.to("cpu")
    
    logger.info("\nresult_path:{}\nfeat_type:{}\nfeat_cof:{}\ndevice:{}\nbatch_size:{}\nclassify_num:{}\n"
            .format(dirpath, args.feat_type, args.feat_cof, model.device, args.batch_size, output_dim))
    logger.info(model)

    model_init(model, init_method="kaiming")
    load_object = Load_Data(phones, args)
    load_object.dataload()
 
    train(model, load_object, args, phones, logger)
    # Tear down the process group
    dist.destroy_process_group()
if __name__ == "__main__":
    main()
