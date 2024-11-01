import torch
import torch.nn as nn
#test(2
import os, sys
import numpy as np
import logging
import json
sys.path.insert(0, "./")
import torch.optim.lr_scheduler as lr_sch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from fqdd.utils.feature import get_feats
from fqdd.prepare_data.aishell_prepare_data import prepare_data_json
from fqdd.utils.lang import create_phones, read_phones
from fqdd.utils.load_data import init_dataset_and_dataloader
from fqdd.utils.train_utils import init_optimizer_and_scheduler, init_distributed
from fqdd.utils.argument import parse_arguments
from fqdd.bin.asr.decode import GreedyDecoder, calculate_cer
from fqdd.utils.init_tokenizer import Tokenizers
from fqdd.models.ebranchformer.ebranchformer import EBranchformer
from fqdd.models.check_model import model_init, save_model, reload_model
from fqdd.utils.logger import init_logging
from fqdd.nnets.losses import nll_loss, transducer_loss
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
    batch = pred.size(0)
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
    loss = loss / batch
    return loss


def train(model, train_loader, dev_loader, optimizer, scheduler, configs, args, logger, rank, device):

    epoch_n = configs["max_epoch"]
    
    if args.pretrained:
        # if args.local_rank < 1:
        start_epoch = reload_model(os.path.join(args.result_dir, str(configs["seed"])), model=model, optimizer=optimizer, map_location='cuda:{}'.format(0))
        start_epoch = start_epoch + 1
    else:
        start_epoch = 1
 
    if rank == 0: 
        logger.info("init_lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
    accum_grad = 4
    for epoch in range(start_epoch, epoch_n+1):
        model.train()
        
        if rank == 0:
            logger.info("Epoch {}/{}".format(epoch, epoch_n))
            logger.info("-" * 10)
        
        infos={"loss": 0.0,
            "ctc_loss": 0.0,
            "att_loss": 0.0,
            "th_acc": 0.0
        }
        log_interval=configs["log_interval"]
        if rank == 0:
            print("status: train\t train_load_size:{}".format(len(train_loader)))
        dist.barrier() # 同步训练进程
        for idx, batch_data in enumerate(tqdm(train_loader)):
            # 只做推理，代码不会更新模型状态 
            keys, feats, wav_lengths, targets, target_lens = batch_data
            feats = feats.to(device)
            wav_lengths = wav_lengths.to(device)
            targets = targets.to(device)
            target_lens = target_lens.to(device)
            info_dicts = model(feats, wav_lengths, targets, target_lens) 
            loss = info_dicts["loss"]
            loss.backward()
            
            # output_en = info_dicts["encoder_out"]

            torch.nn.utils.clip_grad_norm_((p for p in model.parameters()), max_norm=50)
            optimizer.step() 
            optimizer.zero_grad()
            scheduler.step()
            infos["loss"] = info_dicts["loss"] + infos["loss"]
            infos["ctc_loss"] = info_dicts["ctc_loss"] + infos["ctc_loss"]
            infos["att_loss"] = info_dicts["att_loss"] + infos["att_loss"]
            infos["th_acc"] = info_dicts["th_acc"] + infos["th_acc"]
            if rank==0 and (idx+1) % log_interval == 0:
                avg_loss = infos["loss"] / (idx+1)
                avg_ctc_loss = infos["ctc_loss"] / (idx+1)
                avg_att_loss = infos["att_loss"] / (idx+1)
                avg_th_acc =  infos["th_acc"] / (idx+1)
                logger.info("Epoch:{}/{}\ttrain:\tloss:{:.2f}\tctc_loss:{:.2f}\tatt_loss:{}\tth_acc:{}".format(epoch, idx+1, avg_loss, avg_ctc_loss, avg_att_loss, avg_th_acc))
        
                save_model(model, optimizer, epoch, os.path.join(args.result_dir, str(configs["seed"])))
        dist.barrier() # 同步测试进程
        with torch.no_grad():
            loss, ctc_loss, att_loss, th_acc = evaluate(model, dev_loader, epoch, configs, logger, rank, device)
            logger.info("Epoch:{}\tDEV:loss:{}\tctc_loss:{}\tatt_loss:{}\tth_acc:{}".format(epoch, loss, ctc_loss, att_loss, th_acc))
            
    
def evaluate(model, eval_loader, epoch, configs, logger, rank, device):
    
    model.eval()
    infos={"loss": 0.0,
            "ctc_loss": 0.0,
            "att_loss": 0.0,
            "th_acc": 0.0
        }
    log_interval=configs["log_interval"]
    for idx, batch_data in enumerate(tqdm(eval_loader)):
        
        keys, feats, wav_lengths, targets, target_lens = batch_data 
        feats = feats.to(device)
        wav_lengths = wav_lengths.to(device)
        targets = targets.to(device)
        target_lens = target_lens.to(device)
        # print(feats.shape)
        info_dicts = model(feats, wav_lengths, targets, target_lens)
        infos["loss"] = info_dicts["loss"] + infos["loss"]
        infos["ctc_loss"] = info_dicts["ctc_loss"] + infos["ctc_loss"]
        infos["att_loss"] = info_dicts["att_loss"] + infos["att_loss"]
        infos["th_acc"] = info_dicts["th_acc"] + infos["th_acc"]
        if rank==0 and (idx+1) % log_interval == 0:
            avg_loss = infos["loss"] / (idx +1)
            avg_ctc_loss = infos["ctc_loss"] / (idx+1)
            avg_att_loss = infos["att_loss"] / (idx+1)
            avg_th_acc =  infos["th_acc"] / (idx+1)
            logger.info("Epoch:{}/{}\tDEV:\tloss:{:.2f}\tctc_loss:{:.2f}\tatt_loss:{}\tth_acc:{}".format(epoch, idx+1, avg_loss, avg_ctc_loss, avg_att_loss, avg_th_acc))

    loss = infos["loss"] / (idx +1)
    ctc_loss = infos["ctc_loss"] / (idx+1)
    att_loss = infos["att_loss"] / (idx+1)
    th_acc =  infos["th_acc"] / (idx+1)
    return loss, ctc_loss, att_loss, th_acc

def main():

    args = parse_arguments()
    configs = json.load(open(args.config, 'r', encoding="utf-8"))
    # configs = reload_configs(args, configs)
     

    configs["init_infos"] = {}
    # prepare data
    dirpath = os.path.join(args.result_dir, str(configs["seed"]))
    # prepare_data_json(args.data_folder, dirpath)
    
    tokenizer = Tokenizers(configs["data"].get("train_file"))
    
    _, _, rank = init_distributed(args)
    train_set, train_loader, train_sampler, dev_set, dev_loader = init_dataset_and_dataloader(configs["data"],
                                                                                              tokenizer=tokenizer,
                                                                                              seed=configs["seed"]
                                                                                              )
    configs["model"]["vocab_size"] = tokenizer.vocab_size()
    logger = init_logging("train", dirpath)
    model_conf = configs["model"]
    
    # model = Encoder_Decoer(output_dim, feat_shape=[args.batch_size, args.max_during*100, input_dim], output_size=1024)
    model = EBranchformer(model_conf)
    model, optimizer, scheduler = init_optimizer_and_scheduler(configs, model)
    if rank == 0: 
        logger.info(model)
        print(args)
    device = args.device
    model_init(model, init_method="kaiming")
    model.to(device) 
    train(model, train_loader, dev_loader, optimizer, scheduler, configs, args, logger, rank, device)
    # Tear down the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
