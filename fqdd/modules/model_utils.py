import torch
import os
import logging
import re
import json
import datetime

from fqdd.modules.attentions import MultiHeadedCrossAttention, RelPositionMultiHeadedAttention, MultiHeadedAttention
from fqdd.modules.embedings import PositionalEncoding, RelPositionalEncoding, NoPositionalEncoding, \
    WhisperPositionalEncoding, LearnablePositionalEncoding, ParaformerPositinoalEncoding, RopePositionalEncoding
from fqdd.modules.mlps import PositionwiseFeedForward
from fqdd.modules.subsamples import LinearNoSubsampling, EmbedinigNoSubsampling, Conv1dSubsampling2, Conv2dSubsampling4, \
    Conv2dSubsampling2

FQDD_EMBEDDINGS = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
    "abs_pos_paraformer": ParaformerPositinoalEncoding,
    'rope_pos': RopePositionalEncoding,
}

FQDD_MLPS = {
    'position_wise_feed_forward': PositionwiseFeedForward,
}

FQDD_ATTENTIONS = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
    "crossattn": MultiHeadedCrossAttention,
}

FQDD_SUBSAMPLES = {
    "linear": LinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d2": Conv2dSubsampling2,
    "conv2d": Conv2dSubsampling4,
}


def save_state_dict_and_infos(state_dict, path: str, infos=None):
    rank = int(os.environ.get('RANK', 0))
    logging.info('[Rank {}] Checkpoint: save to checkpoint {}'.format(
        rank, path))
    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.json', path)
    if infos is None:
        infos = {}
    infos['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    with open(info_path, 'w') as fout:
        data = json.dumps(infos, indent=4)
        fout.write(data)


def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    '''
    Args:
        infos (dict or None): any info you want to save.

    Parameters
    ----------
    infos
    path
    model
    '''
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    save_state_dict_and_infos(state_dict, path, infos)


def save_model(model: torch.nn.Module, info_dict=None):
    rank = int(os.environ.get('RANK', 0))
    tag = info_dict["tag"]
    model_dir = info_dict["model_dir"]
    save_model_path = os.path.join(model_dir, '{}.pt'.format(tag))
    # save ckpt
    if rank == 0:
        # NOTE(xcsong): For torch_ddp, only rank-0 should call this.
        save_checkpoint(model, save_model_path, info_dict)
        # save yaml
        with open("{}/{}.json".format(model_dir, tag), 'w') as fout:
            data = json.dumps(info_dict, indent=4)
            fout.write(data)


def reload_model(load_dir, model=None, optimizer=None, map_location=None):
    '''
    reload model
    '''
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
    info_path = re.sub('.pt$', '.json', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = json.load(fin)
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
