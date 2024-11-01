import torch
import os, sys
import torch.nn as nn
import time
import logging
import re


def save_model(model, optimizer, epoch, save_dir):
    try:
        # epoch
        os.makedirs(save_dir, exist_ok=True)
        epoch_path = os.path.join(save_dir, 'checkpoint')
        check_epoch = {'checkpoint': epoch}
        torch.save(check_epoch, epoch_path, _use_new_zipfile_serialization=False)

        # model
        model_path = os.path.join(save_dir, 'model.ckpt')
        check_model = {'model': model.state_dict()}
        torch.save(check_model, model_path, _use_new_zipfile_serialization=False)

        # optimizer
        optimizer_path = os.path.join(save_dir, 'optimizer.ckpt')
        check_optimizer = {'optimizer': optimizer.state_dict()}
        torch.save(check_optimizer, optimizer_path, _use_new_zipfile_serialization=False)
    except:
        print('save {}th epoch mode error'.format(epoch))
        return


'''
reload model
'''


def reload_model(load_dir, model=None, optimizer=None, map_location=None):
    print("ready to load pretrain model")

    # load epoch
    if not os.path.exists(os.path.join(load_dir, 'checkpoint')):
        return 0

    try:
        epoch_path = os.path.join(load_dir, 'checkpoint')
        start_epoch = torch.load(epoch_path, map_location=map_location)['checkpoint']
    except:
        print('reload_checkpoint error:\t{}'.format(epoch_path))
        return 0

    # load net params
    try:
        if model:
            model_path = os.path.join(load_dir, 'model.ckpt')
            check_model = torch.load(model_path, map_location=map_location)
            model.load_state_dict(check_model['model'])
    except:
        print('reload_model error:\t{}'.format(model_path))

    # load optimizer
    try:
        if optimizer:
            optimizer_path = os.path.join(load_dir, 'optimizer.ckpt')
            check_optimizer = torch.load(optimizer_path, map_location=map_location)
            optimizer.load_state_dict(check_optimizer['optimizer'])
    except:
        print('reload_optimizer error:\t{}'.format(optimizer_path))
    return start_epoch


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    rank = int(os.environ.get('RANK', 0))
    logging.info('[Rank {}] Checkpoint: loading from checkpoint {}'.format(
        rank, path))
    checkpoint = torch.load(path, map_location='cpu', mmap=True)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint,
                                                          strict=False)
    if rank == 0:
        for key in missing_keys:
            logging.info("missing tensor: {}".format(key))
        for key in unexpected_keys:
            logging.info("unexpected tensor: {}".format(key))
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


def infer_model(load_dir, model):
    # load epoch
    model_path = os.path.join(load_dir, 'model.ckpt')
    try:
        check_model = torch.load(model_path)
        model.load_state_dict(check_model['model'])
        return model
    except:
        print('reload_model, file not exists:\t{}'.format(model_path))
