# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Tsinghua Univ. (authors: Xingchen Song)
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

import re
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from fqdd.utils.common import pad_list


def remove_duplicates_and_blank(hyp: List[int],
                                blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def tokenize_by_bpe_model(sp, txt):
    return _tokenize_by_seg_dic_or_bpe_model(txt, sp=sp, upper=True)


def tokenize_by_seg_dict(seg_dict, txt):
    return _tokenize_by_seg_dic_or_bpe_model(txt,
                                             seg_dict=seg_dict,
                                             upper=False)


def _tokenize_by_seg_dic_or_bpe_model(
        txt,
        sp=None,
        seg_dict=None,
        upper=True,
):
    if sp is None:
        assert seg_dict is not None
    if seg_dict is None:
        assert sp is not None
    tokens = []
    # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    pattern = re.compile(r'([\u4e00-\u9fff])')
    # Example:
    #   txt   = "你好 ITS'S OKAY 的"
    #   chars = ["你", "好", " ITS'S OKAY ", "的"]
    chars = pattern.split(txt.upper() if upper else txt)
    mix_chars = [w for w in chars if len(w.strip()) > 0]
    for ch_or_w in mix_chars:
        # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
        if pattern.fullmatch(ch_or_w) is not None:
            tokens.append(ch_or_w)
        # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
        # encode ch_or_w using bpe_model.
        else:
            if sp is not None:
                for p in sp.encode_as_pieces(ch_or_w):
                    tokens.append(p)
            else:
                for en_token in ch_or_w.split():
                    en_token = en_token.strip()
                    if en_token in seg_dict:
                        tokens.extend(seg_dict[en_token].split(' '))
                    else:
                        tokens.append(en_token)

    return tokens


def add_blank(ys_pad: torch.Tensor, blank: int,
              ignore_id: int) -> torch.Tensor:
    """ Prepad blank for transducer predictor

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        blank (int): index of <blank>

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> blank = 0
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,   4,   5],
                [ 4,  5,  6,  -1,  -1],
                [ 7,  8,  9,  -1,  -1]], dtype=torch.int32)
        >>> ys_in = add_blank(ys_pad, 0, -1)
        >>> ys_in
        tensor([[0,  1,  2,  3,  4,  5],
                [0,  4,  5,  6,  0,  0],
                [0,  7,  8,  9,  0,  0]])
    """
    bs = ys_pad.size(0)
    _blank = torch.tensor([blank],
                          dtype=torch.long,
                          requires_grad=False,
                          device=ys_pad.device)
    _blank = _blank.repeat(bs).unsqueeze(1)  # [bs,1]
    out = torch.cat([_blank, ys_pad], dim=1)  # [bs, Lmax+1]
    return torch.where(out == ignore_id, blank, out)


def add_sos_eos(ys_pad: torch.Tensor, sos: int, eos: int,
                ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add <sos> and <eos> labels.

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        sos (int): index of <sos>
        eos (int): index of <eeos>
        ignore_id (int): index of padding

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + 1)
        ys_out (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, -1, -1],
                [ 7,  8,  9, -1, -1]], dtype=torch.int32)
        >>> ys_in,ys_out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
        >>> ys_in
        tensor([[10,  1,  2,  3,  4,  5],
                [10,  4,  5,  6, 11, 11],
                [10,  7,  8,  9, 11, 11]])
        >>> ys_out
        tensor([[ 1,  2,  3,  4,  5, 11],
                [ 4,  5,  6, 11, -1, -1],
                [ 7,  8,  9, 11, -1, -1]])
    """
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def add_whisper_tokens(special_tokens, ys_pad: torch.Tensor, ignore_id: int,
                       tasks: List[str], no_timestamp: bool, langs: List[str],
                       use_prev: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add whisper-style tokens.

    ([PREV] -> [previous text tokens or hotwords]).optional --
      ┌------------------------------------------------------↲
      ↓
    [sot] -> [language id] -> [transcribe] -> [begin time] -> [text tokens] -> [end time] -> ... -> [eot]    # noqa
        |          |                |-------> [no timestamps] -> [text tokens] ----------------------↑       # noqa
        |          |                                                                                 |       # noqa
        |          |--------> [translate]  -> [begin time] -> [text tokens] -> [end time] -> ... --->|       # noqa
        |                           |-------> [no timestamps] -> [text tokens] --------------------->|       # noqa
        |                                                                                            |       # noqa
        |--> [no speech(VAD)] ---------------------------------------------------------------------->|       # noqa

    Args:
        special_tokens: get IDs of special tokens
        ignore_id (int): index of padding
        no_timestamp (bool): whether to add timestamps tokens
        tasks (List[str]): list of task tags
        langs (List[str]): list of language tags

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + ?)
        ys_out (torch.Tensor) : (B, Lmax + ?)

    """
    assert len(langs) == ys_pad.size(0)
    assert len(tasks) == ys_pad.size(0)
    if use_prev:
        # i.e., hotword list
        _prev = [special_tokens["sot_prev"]]
        # append hotword list to _prev
        # ...
        raise NotImplementedError
    else:
        _prev = []

    _sot = []
    for task, lang in zip(tasks, langs):
        if task == "transcribe":
            task_id = special_tokens["transcribe"]
        elif task == "translate":
            task_id = special_tokens["translate"]
        elif task == "vad":
            task_id = special_tokens["no_speech"]
        else:
            raise NotImplementedError("unsupported task {}".format(task))
        language_id = special_tokens["sot"] + 1 + WHISPER_LANGS.index(lang)
        prefix = _prev + [special_tokens["sot"], language_id, task_id]
        if task == "transcribe" or task == "translate":
            if no_timestamp:
                prefix.append(special_tokens["no_timestamps"])
            else:
                prefix.append(special_tokens["timestamp_begin"])
                # add subsequent tokens
                # ...
                raise NotImplementedError
        elif task == "vad":
            prefix.append(special_tokens["no_speech"])
        else:
            raise NotImplementedError
        prefix = torch.tensor(prefix,
                              dtype=torch.long,
                              requires_grad=False,
                              device=ys_pad.device)
        _sot.append(prefix)

    _eot = torch.tensor([special_tokens["eot"]],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys

    ys_in = [torch.cat([prefix, y], dim=0) for prefix, y in zip(_sot, ys)]
    ys_out = [
        torch.cat([prefix[1:], y, _eot], dim=0) for prefix, y in zip(_sot, ys)
    ]
    return pad_list(ys_in, special_tokens["eot"]), pad_list(ys_out, ignore_id)


def reverse_pad_list(ys_pad: torch.Tensor,
                     ys_lens: torch.Tensor,
                     pad_value: float = -1.0) -> torch.Tensor:
    """Reverse padding for the list of tensors.

    Args:
        ys_pad (tensor): The padded tensor (B, Tokenmax).
        ys_lens (tensor): The lens of token seqs (B)
        pad_value (int): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tokenmax).

    Examples:
        >>> x
        tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]])
        >>> pad_list(x, 0)
        tensor([[4, 3, 2, 1],
                [7, 6, 5, 0],
                [9, 8, 0, 0]])

    """
    r_ys_pad = pad_sequence([(torch.flip(y.int()[:i], [0]))
                             for y, i in zip(ys_pad, ys_lens)], True,
                            pad_value)
    return r_ys_pad
