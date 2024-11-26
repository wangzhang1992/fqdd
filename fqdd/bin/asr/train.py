import datetime
from contextlib import nullcontext

import torch
import os, sys
import json
import torch.distributed as dist

sys.path.insert(0, "./")

from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from fqdd.utils.load_data import init_dataset_and_dataloader
from fqdd.utils.train_utils import init_optimizer_and_scheduler, init_distributed
from fqdd.utils.argument import parse_arguments, reload_configs
from fqdd.text.init_tokenizer import Tokenizers
from fqdd.modules.model_utils import save_model
from fqdd.models.init_model import init_model
from fqdd.utils.logger import init_logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


def train(model, train_loader, dev_loader, optimizer, scheduler, configs, logger, rank, device):
    if rank == 0:
        print("status: train\t train_load_size:{}".format(len(train_loader)))

    log_interval = configs["log_interval"]
    tag = configs["init_infos"].get("tag", "init")
    start_epoch = configs["init_infos"].get('epoch', 0) + int("epoch_" in tag)
    epoch_n = configs["max_epoch"]

    clip = configs["model"]["grad_clip"]
    accum_grad = configs["accumulation_steps"]
    train_engine = configs["dist_conf"]["train_engine"]

    if rank == 0:
        logger.info("init_lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
    final_epoch = None

    for epoch in range(start_epoch, epoch_n):

        if rank == 0:
            logger.info("Epoch {}/{}".format(epoch, epoch_n))
            logger.info("-" * 50)

        infos = {"loss": [],
                 "ctc_loss": [],
                 "att_loss": [],
                 "th_acc": []
                 }
        # 每一次新的epoch，重新打乱数据
        train_loader.sampler.set_epoch(epoch)
        dist.barrier()  # 同步训练进程:
        group_join = dist.new_group(
            backend="gloo", timeout=datetime.timedelta(seconds=30))
        model.train()

        for idx, batch_data in enumerate(tqdm(train_loader)):
            # 只做推理，代码不会更新模型状态
            keys, feats, wav_lengths, targets, target_lens = batch_data
            feats = feats.to(device)
            wav_lengths = wav_lengths.to(device)
            targets = targets.to(device)
            target_lens = target_lens.to(device)

            context = None
            # Disable gradient synchronizations across DDP processes.
            # Within this context, gradients will be accumulated on module
            # variables, which will later be synchronized.
            if train_engine in ["torch_ddp", "torch_fsdp"] and (idx + 1) % accum_grad != 0:
                context = model.no_sync
            # Used for single gpu training and DDP gradient synchronization
            # processes.
            else:
                context = nullcontext
            with context():
                batch_infos = model(feats, wav_lengths, targets, target_lens)

                assert train_engine in ["torch_ddp", "torch_fsdp"]
                scaled_loss = batch_infos["loss"] / accum_grad
                scaled_loss.backward()

            if (idx + 1) % accum_grad == 0:
                if train_engine == "torch_ddp":
                    grad_norm = clip_grad_norm_((p for p in model.parameters()), max_norm=clip)
                else:
                    grad_norm = model.clip_grad_norm_(clip)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            infos["loss"].append(batch_infos["loss"].item())
            infos["ctc_loss"].append(batch_infos["ctc_loss"].item())
            infos["att_loss"].append(batch_infos["att_loss"].item())
            infos["th_acc"].append(batch_infos["th_acc"].item())
            if rank == 0 and (idx + 1) % log_interval == 0 and (idx + 1) % accum_grad == 0:
                interval_loss = sum(infos["loss"][-log_interval:]) / log_interval
                interval_ctc_loss = sum(infos["ctc_loss"][-log_interval:]) / log_interval
                interval_att_loss = sum(infos["att_loss"][-log_interval:]) / log_interval
                interval_th_acc = sum(infos["th_acc"][-log_interval:]) / log_interval
                logger.info(
                    "Epoch:{}/{}\ttrain:\tloss:{:.4f}\tctc_loss:{:.4f}\tatt_loss:{:.4f}\tth_acc:{:.4f}\tlr:{:.6f}".format(
                        epoch,
                        idx + 1,
                        interval_loss,
                        interval_ctc_loss,
                        interval_att_loss,
                        interval_th_acc,
                        optimizer.param_groups[0]["lr"]))

        if rank == 0:
            train_loss = sum(infos["loss"]) / (idx + 1)
            train_ctc_loss = sum(infos["ctc_loss"]) / (idx + 1)
            train_att_loss = sum(infos["att_loss"]) / (idx + 1)
            train_th_acc = sum(infos["th_acc"]) / (idx + 1)
            logger.info(
                "Epoch:{}\ttrain:\tloss:{:.4f}\tctc_loss:{:.4f}\tatt_loss:{:.4f}\tth_acc:{:.4f}".format(epoch,
                                                                                                        train_loss,
                                                                                                        train_ctc_loss,
                                                                                                        train_att_loss,
                                                                                                        train_th_acc)
            )

        dist.destroy_process_group(group_join)
        dist.barrier()  # 同步测试进程
        with torch.no_grad():
            loss, ctc_loss, att_loss, th_acc = evaluate(model, dev_loader, epoch, configs, logger, rank, device)
            logger.info(
                "Epoch:{}\tCV:loss:{:.4f}\tctc_loss:{:.4f}\tatt_loss:{:.4f}\tth_acc:{:.4f}".format(epoch, loss,
                                                                                                   ctc_loss, att_loss,
                                                                                                   th_acc)
            )
        info_dict = {
            "epoch": epoch,
            "save_time": datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            "tag": "epoch_{}".format(epoch),
            "step": scheduler.last_epoch,
            "result_dict": {
                "loss": loss,
                "ctc_loss": ctc_loss,
                "att_loss": att_loss,
                "th_acc": th_acc
            },
            **configs
        }
        save_model(model, info_dict)
        final_epoch = epoch

    if final_epoch is not None and rank == 0:
        final_model_path = os.path.join(configs["model_dir"], 'final.pt')
        os.remove(final_model_path) if os.path.exists(
            final_model_path) else None
        os.symlink('{}.pt'.format(final_epoch), final_model_path)


def evaluate(model, eval_loader, epoch, configs, logger, rank, device):
    model.eval()
    infos = {"loss": [],
             "ctc_loss": [],
             "att_loss": [],
             "th_acc": []
             }

    log_interval = configs["log_interval"]
    for idx, batch_data in enumerate(tqdm(eval_loader)):

        keys, feats, wav_lengths, targets, target_lens = batch_data
        feats = feats.to(device)
        wav_lengths = wav_lengths.to(device)
        targets = targets.to(device)
        target_lens = target_lens.to(device)
        # print(feats.shape)
        batch_infos = model(feats, wav_lengths, targets, target_lens)
        infos["loss"].append(batch_infos["loss"].item())
        infos["ctc_loss"].append(batch_infos["ctc_loss"].item())
        infos["att_loss"].append(batch_infos["att_loss"].item())
        infos["th_acc"].append(batch_infos["th_acc"].item())
        if rank == 0 and (idx + 1) % log_interval == 0:
            interval_loss = sum(infos["loss"][-log_interval:]) / log_interval
            interval_ctc_loss = sum(infos["ctc_loss"][-log_interval:]) / log_interval
            interval_att_loss = sum(infos["att_loss"][-log_interval:]) / log_interval
            interval_th_acc = sum(infos["th_acc"][-log_interval:]) / log_interval
            logger.info(
                "Epoch:{}/{}\tCV:\tloss:{:.4f}\tctc_loss:{:.4f}\tatt_loss:{:.4f}\tth_acc:{:.4f}".format(epoch, idx + 1,
                                                                                                        interval_loss,
                                                                                                        interval_ctc_loss,
                                                                                                        interval_att_loss,
                                                                                                        interval_th_acc)
            )
    cv_loss = sum(infos["loss"]) / (idx + 1)
    cv_ctc_loss = sum(infos["ctc_loss"]) / (idx + 1)
    cv_att_loss = sum(infos["att_loss"]) / (idx + 1)
    cv_th_acc = sum(infos["th_acc"]) / (idx + 1)
    return cv_loss, cv_ctc_loss, cv_att_loss, cv_th_acc


def main():
    args = parse_arguments()
    configs = json.load(open(args.train_config, 'r', encoding="utf-8"))

    configs = reload_configs(args, configs)

    configs["init_infos"] = {}
    # prepare data
    logger = init_logging("train", configs["model_dir"])
    # prepare_data_json(args.data_folder, configs["model_dir"])

    tokenizer = Tokenizers(configs)

    _, _, rank = init_distributed(args)
    train_set, train_loader, train_sampler, dev_set, dev_loader = init_dataset_and_dataloader(args,
                                                                                              configs,
                                                                                              tokenizer=tokenizer,
                                                                                              seed=configs["seed"]
                                                                                              )
    configs["model"]["vocab_size"] = tokenizer.vocab_size()

    model, configs = init_model(args, configs)
    model, optimizer, scheduler = init_optimizer_and_scheduler(configs, model)

    # Save checkpoints
    save_model(model,
               info_dict={
                   "save_time": datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                   "tag": "init",
                   **configs
               })
    if rank == 0:
        logger.info(model)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info('the number of model params: {:,d}'.format(num_params))
    device = args.device
    model.to(device)
    train(model, train_loader, dev_loader, optimizer, scheduler, configs, logger, rank, device)
    # Tear down the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
