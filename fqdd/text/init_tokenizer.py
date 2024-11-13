import os
import json
import torch

from typing import List


class Tokenizers:

    def __init__(self,
                 config,
                 token_type: str = "char",
                 special_tokens=None,
                 unk: str = "<unk>",
                 connect_symbol: str = ''
                 ):

        self.unk = unk
        self.dict_path = config["tokenizer_conf"]["symbol_table_path"]
        self.split_with_space = config["tokenizer_conf"]["split_with_space"]
        self.special_tokens = special_tokens
        self.connect_symbol = connect_symbol
        self.ch2ids_dict, self.ids2ch_dict = self.read_file()
        self.blank_id = self.ch2ids_dict["<blank>"]

    def read_file(self):

        print(self.dict_path)
        assert os.path.exists(self.dict_path) is True

        ch2ids_dict = {k: int(v) for k, v in [kv.split(" ") for kv in open(self.dict_path, 'r').readlines()]}
        ids2ch_dict = {v: k for k, v in ch2ids_dict.items()}

        return ch2ids_dict, ids2ch_dict

    def text2tokens(self, line: str) -> List[str]:
        line = line.strip()

        parts = [line]

        tokens = []
        for part in parts:
            if self.split_with_space:
                part = part.split(" ")
            for ch in part:
                if ch == ' ':
                    ch = "▁"
                tokens.append(ch)
        return tokens

    def tokens2text(self, tokens: List[str]) -> str:
        return self.connect_symbol.join(tokens)

    def tokens2ids(self, tokens: List[str]) -> List[int]:

        ids = []
        for ch in tokens:
            if ch in self.ch2ids_dict.keys():
                ids.append(self.ch2ids_dict[ch])
            else:
                ids.append(self.ch2ids_dict[self.unk])

        return ids

    def id2tokens(self, ids: List[int]) -> List[str]:
        content = ""
        if len(ids) == 0:
            pass
        else:
            content = [self.ids2ch_dict[w] for w in ids]
        return content

    def vocab_size(self):
        return len(self.ch2ids_dict)


'''
tokenizer = Tokenizers("data/train/data.list")
'''
