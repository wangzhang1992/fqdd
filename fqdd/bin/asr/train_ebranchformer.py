import datetime

import torch
import os, sys
import json
import torch.distributed as dist

sys.path.insert(0, "./")

from tqdm import tqdm
from fqdd.utils.load_data import init_dataset_and_dataloader
from fqdd.utils.train_utils import init_optimizer_and_scheduler, init_distributed
from fqdd.utils.argument import parse_arguments, reload_configs
from fqdd.text.init_tokenizer import Tokenizers
from fqdd.models.model_utils import save_model
from fqdd.models.init_model import init_model
from fqdd.utils.logger import init_logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


def train(model, train_loader, dev_loader, optimizer, scheduler, configs, logger, rank, device):

    # if args.pretrained:
    #     # if args.local_rank < 1:
    #     start_epoch = reload_model(os.path.join(args.result_dir, str(configs["seed"])), model=model,
    #                                optimizer=optimizer, map_location='cuda:{}'.format(0))
    #     start_epoch = start_epoch + 1
    # else:
    #     start_epoch = 1
    tag = configs["init_infos"].get("tag", "init")
    start_epoch = configs["init_infos"].get('epoch', 0) + int("epoch_" in tag)
    epoch_n = configs["max_epoch"]
    if rank == 0:
        logger.info("init_lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
    accum_grad = 4
    for epoch in range(start_epoch, epoch_n):
        model.train()

        if rank == 0:
            logger.info("Epoch {}/{}".format(epoch, epoch_n))
            logger.info("-" * 50)

        infos = {"loss": 0.0,
                 "ctc_loss": 0.0,
                 "att_loss": 0.0,
                 "th_acc": 0.0
                 }
        log_interval = configs["log_interval"]
        if rank == 0:
            print("status: train\t train_load_size:{}".format(len(train_loader)))
        dist.barrier()  # 同步训练进程
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
            infos["loss"] = info_dicts["loss"].item() + infos["loss"]
            infos["ctc_loss"] = info_dicts["ctc_loss"].item() + infos["ctc_loss"]
            infos["att_loss"] = info_dicts["att_loss"] + infos["att_loss"]
            infos["th_acc"] = info_dicts["th_acc"].item() + infos["th_acc"]
            if rank == 0 and (idx + 1) % log_interval == 0:
                avg_loss = infos["loss"] / (idx + 1)
                avg_ctc_loss = infos["ctc_loss"] / (idx + 1)
                avg_att_loss = infos["att_loss"] / (idx + 1)
                avg_th_acc = infos["th_acc"] / (idx + 1)
                logger.info(
                    "Epoch:{}/{}\ttrain:\tloss:{:.2f}\tctc_loss:{:.2f}\tatt_loss:{}\tth_acc:{}".format(epoch, idx + 1,
                                                                                                       avg_loss,
                                                                                                       avg_ctc_loss,
                                                                                                       avg_att_loss,
                                                                                                       avg_th_acc)
                )

        dist.barrier()  # 同步测试进程
        with torch.no_grad():
            loss, ctc_loss, att_loss, th_acc = evaluate(model, dev_loader, epoch, configs, logger, rank, device)
            logger.info(
                "Epoch:{}\tDEV:loss:{}\tctc_loss:{}\tatt_loss:{}\tth_acc:{}".format(epoch, loss, ctc_loss, att_loss,
                                                                                    th_acc)
            )
        info_dict = {
            "epoch": epoch,
            "save_time": datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            "tag": "epoch_{}".format(epoch),
            "step": scheduler.last_epoch,
            "result_dict": {
                "loss":loss,
                "ctc_loss": ctc_loss,
                "att_loss": att_loss,
                "th_acc": th_acc
            },
            **configs
        }
        save_model(model, info_dict)


def evaluate(model, eval_loader, epoch, configs, logger, rank, device):
    model.eval()
    infos = {"loss": 0.0,
             "ctc_loss": 0.0,
             "att_loss": 0.0,
             "th_acc": 0.0
             }
    log_interval = configs["log_interval"]
    for idx, batch_data in enumerate(tqdm(eval_loader)):

        keys, feats, wav_lengths, targets, target_lens = batch_data
        feats = feats.to(device)
        wav_lengths = wav_lengths.to(device)
        targets = targets.to(device)
        target_lens = target_lens.to(device)
        # print(feats.shape)
        info_dicts = model(feats, wav_lengths, targets, target_lens)
        infos["loss"] = info_dicts["loss"].item() + infos["loss"]
        infos["ctc_loss"] = info_dicts["ctc_loss"].item() + infos["ctc_loss"]
        infos["att_loss"] = info_dicts["att_loss"].item() + infos["att_loss"]
        infos["th_acc"] = info_dicts["th_acc"].item() + infos["th_acc"]
        if rank == 0 and (idx + 1) % log_interval == 0:
            avg_loss = infos["loss"] / (idx + 1)
            avg_ctc_loss = infos["ctc_loss"] / (idx + 1)
            avg_att_loss = infos["att_loss"] / (idx + 1)
            avg_th_acc = infos["th_acc"] / (idx + 1)
            logger.info("Epoch:{}/{}\tDEV:\tloss:{:.2f}\tctc_loss:{:.2f}\tatt_loss:{}\tth_acc:{}".format(epoch, idx + 1,
                                                                                                         avg_loss,
                                                                                                         avg_ctc_loss,
                                                                                                         avg_att_loss,
                                                                                                         avg_th_acc))

    loss = infos["loss"] / (idx + 1)
    ctc_loss = infos["ctc_loss"] / (idx + 1)
    att_loss = infos["att_loss"] / (idx + 1)
    th_acc = infos["th_acc"] / (idx + 1)
    return loss, ctc_loss, att_loss, th_acc


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
    device = args.device
    model.to(device)
    train(model, train_loader, dev_loader, optimizer, scheduler, configs, logger, rank, device)
    # Tear down the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

