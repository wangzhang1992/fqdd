"""A combination of Convolutional, Recurrent, and Fully-connected networks.

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Ju-Chieh Chou 2020
 * Titouan Parcollet 2020
 * Abdel 2020
"""
import os, sys, torch
import torch.nn as nn
sys.path.insert(0, './')
from thop import profile
from script.nnets.containers import Sequential
from script.nnets.CNN import Conv2d
from script.nnets.dropout import Dropout2d
from script.nnets.normalization import LayerNorm, BatchNorm1d
from script.nnets.pooling import Pooling1d, Pooling2d
from script.nnets.RNN import LiGRU, LSTM, GRU
from script.nnets.linear import Linear
from script.nnets.embedding import Embedding
from transformers import Wav2Vec2Model


class PreTrain_Wav2Vec(nn.Module):

    def __init__(
        self, input_size, output_size, activation=nn.LeakyReLU, dropout=0.15
    ):
        super(PreTrain_Wav2Vec, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("./pretrain_model_asr")
        self.bn = nn.BatchNorm1d(input_size)
        self.drop = nn.Dropout(p=dropout)
        self.act = activation()
        self.out = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        encoder_out = self.wav2vec(x)
        last_hidden_state = encoder_out.last_hidden_state.transpose(1, -1)
        # print("ptw_f:{}".format(last_hidden_state.shape))
        x = self.bn(last_hidden_state)
        x = self.act(x)
        x = self.drop(x)
        # print(x.shape)
        x = self.out(x.transpose(1, -1))
        # print(x.shape)
        return x
        
class LM_Decoder(nn.Module):
    def __init__(
        self,
        classify,
        consider_as_one_hot=False,
        embedding_dim: int=512,
        output_size: int=512,
        dropout=0.15,
        blank_id=0,
     ):
        super(LM_Decoder, self).__init__()
        self.embedding = Embedding(num_embeddings=classify, embedding_dim=512, consider_as_one_hot=False)
        self.rnn1 = nn.LSTM(input_size=embedding_dim, hidden_size=output_size, num_layers=2, batch_first=True, dropout=0.15, bidirectional=False)
        self.l1 = nn.Linear(output_size, output_size, bias=True)
        self.batchnorm_1 = nn.LayerNorm(output_size)
        self.act_1 = nn.LeakyReLU()
        self.drop_1 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.embedding(x)
        # print(x.shape)
        x, _ = self.rnn1(x)
        # print(x.shape)
        x = self.l1(x)
        x = self.batchnorm_1(x)
        # print(x.shape)
        x = self.act_1(x)
        x = self.drop_1(x)
        return x


class Encoder_Decoer(nn.Module):
     
    def __init__(
        self,
        output_classify,
        feat_size=None,
        output_size: int=1024,
        embedding_dim: int=512,
        dropout=0.15
    ):
        super(Encoder_Decoer, self).__init__()
        
        self.encoder = PreTrain_Wav2Vec(output_size, output_size)
        self.decoder = LM_Decoder(output_classify, embedding_dim=embedding_dim, output_size=output_size)
        #self.decoder = LM_Decoder(output_classify)
        self.en_l = nn.Linear(output_size, output_classify, bias=False)
        self.de_l = nn.Linear(output_size, output_classify, bias=False)
        self.join_l = nn.Linear(output_size, output_classify, bias=False)     
        
    def forward(self, feats, targets_bos=None):
        output_de = None
        output_join = None
        # print(feats.shape)
        en = self.encoder(feats)
        output_en = self.en_l(en)
        en = en.unsqueeze(1)
        if targets_bos != None:
            de = self.decoder(targets_bos)
            output_de = self.de_l(de)
            de = de.unsqueeze(2)
            join = en + de
            output_join = self.join_l(join)
        #return output_en, output_de, output_join
        return output_en, output_de
'''
torch.manual_seed(2024)
feats = torch.randn(2, 29050).to("cuda:0")
targets = torch.LongTensor([[1,21,19,19,28,40],[4,89,19, 22,28,89]]).to("cuda:0")
print("input_feats.shape:{}".format(feats.shape))
print("input_targets.shape:{}".format(targets.shape))
net = Encoder_Decoer(4078, output_size=1024).to("cuda:0")
print(net)
res = net(feats, targets)
print(res[0].size(), res[1].size())
'''
