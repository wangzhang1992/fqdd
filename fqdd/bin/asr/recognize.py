# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import json
import logging
import os
import sys

sys.path.insert(0, "./")
import torch
from torch.utils.data import DataLoader

from fqdd.utils.argument import reload_configs
from fqdd.models.init_model import init_model
from fqdd.utils.load_data import Dataload
from fqdd.text.init_tokenizer import Tokenizers
from fqdd.utils.load_data import collate_fn


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--configs', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--device',
                        type=str,
                        default="cpu",
                        choices=["cpu", "npu", "cuda"],
                        help='accelerator to use')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp32',
                        choices=['fp16', 'fp32', 'bf16'],
                        help='model\'s dtype')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--length_penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--blank_penalty',
                        type=float,
                        default=0.0,
                        help='blank penalty')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--modes',
                        nargs='+',
                        help="""decoding mode, support the following:
                             attention
                             ctc_greedy_search
                             ctc_prefix_beam_search
                             attention_rescoring
                             rnnt_greedy_search
                             rnnt_beam_search
                             rnnt_beam_attn_rescoring
                             ctc_beam_td_attn_rescoring
                             hlg_onebest
                             hlg_rescore
                             paraformer_greedy_search
                             paraformer_beam_search""")
    parser.add_argument('--search_ctc_weight',
                        type=float,
                        default=1.0,
                        help='ctc weight for nbest generation')
    parser.add_argument('--search_transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for nbest generation')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for rescoring weight in \
                                  attention rescoring decode mode \
                              ctc weight for rescoring weight in \
                                  transducer attention rescore decode mode')

    parser.add_argument('--transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for rescoring weight in '
                             'transducer attention rescore mode')
    parser.add_argument('--attn_weight',
                        type=float,
                        default=0.0,
                        help='attention weight for rescoring weight in '
                             'transducer attention rescore mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    parser.add_argument('--word',
                        default='',
                        type=str,
                        help='word file, only used for hlg decode')
    parser.add_argument('--hlg',
                        default='',
                        type=str,
                        help='hlg file, only used for hlg decode')
    parser.add_argument('--lm_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--r_decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    if args.gpu != -1:
        # remain the original usage of gpu
        args.device = "cuda"
    if "cuda" in args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    configs = json.load(open(args.configs, 'r', encoding="utf-8"))
    # configs = reload_configs(args, configs)

    test_conf = copy.deepcopy(configs["data_conf"])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['sort'] = False
    test_conf['shuffle'] = False
    test_conf["augment"]["speed_perturb"] = False
    test_conf["augment"]["wav_distortion"] = False
    test_conf["augment"]["add_reverb"] = False
    test_conf["augment"]["add_noise"] = False
    test_conf["augment"]['spec_aug'] = False
    test_conf["augment"]['spec_sub'] = False
    test_conf["augment"]['spec_trim'] = False
    test_conf["filter"] = False

    if test_conf["feat_type"] == "fbank":
        test_conf['fbank_conf']['dither'] = 0.0
    elif test_conf["feat_type"] == 'mfcc':
        test_conf["mfcc_conf"]['dither'] = 0.0
    test_conf['batch_size'] = args.batch_size

    tokenizer = Tokenizers(configs)
    configs["model"]["vocab_size"] = tokenizer.vocab_size()

    test_dataset = Dataload(args.test_data,
                            test_conf,
                            tokenizer
                            )

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=test_conf.get("batch_size", 1),
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn,
                                  )

    # Init asr model from configs
    args.jit = False
    model, configs = init_model(args, configs)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    dtype = torch.float32
    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    logging.info("compute dtype is {}".format(dtype))

    # blank_id = tokenizer.tokens2ids("<blank>")
    blank_id = tokenizer.blank_id
    logging.info("blank_id is {}".format(blank_id))

    # TODO(Dinghao Zhou): Support RNN-T related decoding
    # TODO(Lv Xiang): Support k2 related decoding
    # TODO(Kaixun Huang): Support context graph
    files = {}
    for mode in args.modes:
        dir_name = os.path.join(args.result_dir, mode)
        os.makedirs(dir_name, exist_ok=True)
        file_name = os.path.join(dir_name, 'text')
        files[mode] = open(file_name, 'w')
    max_format_len = max([len(mode) for mode in args.modes])

    with torch.cuda.amp.autocast(enabled=True,
                                 dtype=dtype,
                                 cache_enabled=False):
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data_loader):
                keys, feats, feats_lengths, targets, target_lengths = batch
                feats = feats.to(device)
                feats_lengths = feats_lengths.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                results = model.decode(
                    feats,
                    feats_lengths,
                    beam_size=args.beam_size,
                    methods=args.modes,
                    blank_id=blank_id,
                    blank_penalty=args.blank_penalty
                )

                for mode, hyps in results.items():
                    print(mode, hyps)
                    for i, key in enumerate(keys):
                        assert hyps[0] is not list
                        tokens = "".join(tokenizer.id2tokens(hyps[i]))
                        line = '{} {}\n'.format(key, tokens)
                        logging.info('{} {} {}'.format(mode, key, tokens))
                        files[mode].write(line)
        for mode, f in files.items():
            f.close()


if __name__ == '__main__':
    main()
