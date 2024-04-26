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

# from apex import amp
from tqdm import tqdm
from script.utils.feature import get_feats
#from script.prepare_data.aishell_prepare_data import prepare_data
from script.prepare_data.thch30_prepare_data import prepare_data
from script.utils.lang import create_phones, read_phones
from script.utils.load_data import Load_Data
from script.utils.argument import parse_arguments
from script.models.las import LAS
from script.asr.decode import GreedyDecoder, calculate_cer
from script.models.check_model import model_init, save_model, reload_model
from script.utils.optimizers import adam_optimizer, sgd_optimizer, scheduler, warmup_lr
from script.utils.logger import init_logging
from script.nnets.losses import calculate_celoss
from script.utils.edit_distance import Static_Cer


#matplotlib.use('Qt5Agg')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

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


def train(model, load_object, args, phones, logger, device):

    min_loss = 1000
    lr_cer = 100
    slice_len = 5    
    epoch_n = args.epoch_n

    train_sampler = load_object.train_sampler
    train_load = load_object.train_load
    dev_load = load_object.dev_load

    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean') 
    optimizer = adam_optimizer(model, args.lr)
    #optimizer = sgd_optimizer(model, args.lr)
    warm_up = warmup_lr(args.lr, (epoch_n+1)*len(train_load))
    #scheduler_lr = scheduler(optimizer, patience=0, cooldown=0)
    
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
    
    for epoch in range(start_epoch, epoch_n+1):

        model.train()
        if args.is_distributed:
            train_sampler.set_epoch(epoch)
        if args.local_rank < 1:
            logger.info("Epoch {}/{}".format(epoch, epoch_n))
            logger.info("-" * 10)
        stat_cer = Static_Cer()
        stat_loss = 0
        if args.local_rank < 1:
            print("status: train\t train_load_size:{}".format(len(train_load)))
        for idx, data in enumerate(tqdm(train_load)):

            feats, targets, targets_bos, targets_eos, wav_lengths, target_lens, target_os_lens = [item.to(device) for item in data]
  
            output_en, output_de = model(feats, targets_bos) 
            output_en = torch.nn.functional.log_softmax(output_en, dim=len(output_en.shape)-1)
            output_de = torch.nn.functional.log_softmax(output_de, dim=len(output_de.shape)-1)
            ctcloss = calculate_loss(ctc_loss, output_en, targets, wav_lengths, target_lens)
            celoss = calculate_celoss(output_de, targets_eos, ignore_index=0, reduction="mean")
            loss = 0.7*ctcloss + celoss
            torch.distributed.barrier()
            optimizer.zero_grad()
            loss.backward()
            # 2.梯度裁剪试试
  
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters()), max_norm=2)
            optimizer.step()
            
            warm_up(optimizer)
            targ, pred = GreedyDecoder(output_en, targets, wav_lengths, target_lens, phones)
            cer = stat_cer.batch_cer(targ, pred)

            stat_loss += loss.item()
            loss = loss.item()
            if idx !=0 and (idx+1)%20 ==0:
                logger.info("batchs:{}   Loss:{:.2f}   CER:{:.2f}".format(idx+1, loss, cer))
            torch.cuda.empty_cache() 
        cer, corr, det, ins, sub = stat_cer.static_cer()
        loss = stat_loss / len(train_load)
        #scheduler_lr.step(loss)
        if args.local_rank < 1:
            logger.info("Epoch:{}, loss:{:.2f}, cer:{:.2f}, lr:{:.2e}, corr:{}, det:{}, ins:{}, sub:{}".format(epoch, loss, cer, optimizer.state_dict()['param_groups'][0]['lr'], corr, det, ins, sub))
            save_model(model, optimizer, epoch, os.path.join(args.result_dir, str(args.seed), 'save', 'AM'))

        with torch.no_grad():
            loss, cer, corr, det, ins, sub = evaluate(model, dev_load, ctc_loss, args, phones, device)

            logger.info("DEV:  loss:{:.2f}, cer:{:.2f}, corr:{}, det:{}, ins:{}, sub:{}".format(loss, cer, corr, det, ins, sub))
        

def evaluate(model, eval_load, ctc_loss, args, phones, device):
    
    model.eval()

    dev_cer = Static_Cer()
    dloss = 0
    dcer = 0

    for idx, data in enumerate(tqdm(eval_load)):
        feats, targets, targets_bos, targets_eos, wav_lengths, target_lens, target_os_lens = [item.to(device) for item in data]

        output_en, output_de = model(feats, targets_bos)
        output_en = torch.nn.functional.log_softmax(output_en, dim=len(output_en.shape)-1)
        output_de = torch.nn.functional.log_softmax(output_de, dim=len(output_de.shape)-1)

        ctcloss = calculate_loss(ctc_loss, output_en, targets, wav_lengths, target_lens)
        celoss = calculate_celoss(output_de, targets_eos, ignore_index=0, reduction="mean")
        loss = 0.7*ctcloss + celoss

        torch.distributed.barrier()
        targ, pred = GreedyDecoder(output_en, targets, wav_lengths, target_lens, phones) 
        cer = dev_cer.batch_cer(targ, pred)
        loss = loss.detach()
        dloss += loss.item()
        torch.cuda.empty_cache()
    loss = dloss / len(eval_load)
    cer, corr, det, ins, sub = dev_cer.static_cer()
    return loss, cer, corr, det, ins, sub

def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    #prepare data
    dirpath = os.path.join(args.result_dir, str(args.seed))
    prepare_data(args.data_folder, dirpath)
    phones = create_phones(dirpath)

    logger = init_logging("train", dirpath)

    if args.feat_type == 'mfcc':
        input_dim = args.feat_cof * 3
    elif args.feat_type == 'fbank':
        input_dim = args.feat_cof

    output_dim = len(phones)

    model = LAS(input_dim, output_dim, args)  

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
        if args.is_distributed:
            if args.local_rank != -1:
                torch.cuda.set_device(args.local_rank)
                device = torch.device(args.local_rank)
        else:
            device = torch.device(args.cuda_device)
    else:
        device = torch.device('cpu')
    
    if args.is_distributed:
        dist.init_process_group(backend='nccl', init_method=args.host, rank=args.local_rank, world_size=args.world_size)

    model = model.to(device)
    
    if args.is_distributed:
        num_gpus = torch.cuda.device_count()
        print(num_gpus, args.local_rank)
        if num_gpus > 1:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    if args.local_rank <1:
        logger.info("\nresult_path:{}\nfeat_type:{}\nfeat_cof:{}\ndevice:{}\nbatch_size:{}\nclassify_num:{}\n"
            .format(dirpath, args.feat_type, args.feat_cof, device, args.batch_size, output_dim))
        logger.info(model)

    model_init(model, init_method="kaiming")
    load_object = Load_Data(phones, args)
    load_object.dataload()
 
    train(model, load_object, args, phones, logger, device)

if __name__ == "__main__":
    main()
