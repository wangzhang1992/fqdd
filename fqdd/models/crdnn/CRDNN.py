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
import torch.nn.functional as F
import math
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

class Encoder(Sequential):
    """This model is a combination of CNNs, RNNs, and DNNs.

    This model expects 3-dimensional input [batch, time, feats] and
    by default produces output of the size [batch, time, dnn_neurons].

    One exception is if ``using_2d_pooling`` or ``time_pooling`` is True.
    In this case, the time dimension will be downsampled.

    Arguments
    ---------
    input_size : int
        The length of the expected input at the third dimension.
    input_shape : tuple
        While input_size will suffice, this option can allow putting
        CRDNN into a sequential with other classes.
    activation : torch class
        A class used for constructing the activation layers for CNN and DNN.
    dropout : float
        Neuron dropout rate as applied to CNN, RNN, and DNN.
    cnn_blocks : int
        The number of convolutional neural blocks to include.
    cnn_channels : list of ints
        A list of the number of output channels for each CNN block.
    cnn_kernelsize : tuple of ints
        The size of the convolutional kernels.
    time_pooling : bool
        Whether to pool the utterance on the time axis before the RNN.
    time_pooling_size : int
        The number of elements to pool on the time axis.
    time_pooling_stride : int
        The number of elements to increment by when iterating the time axis.
    using_2d_pooling: bool
        Whether using a 2D or 1D pooling after each CNN block.
    inter_layer_pooling_size : list of ints
        A list of the pooling sizes for each CNN block.
    rnn_class : torch class
        The type of RNN to use in CRDNN network (LiGRU, LSTM, GRU, RNN)
    rnn_layers : int
        The number of recurrent RNN layers to include.
    rnn_neurons : int
        Number of neurons in each layer of the RNN.
    rnn_bidirectional : bool
        Whether this model will process just forward or in both directions.
    rnn_re_init : bool,
        If True, an orthogonal initialization will be applied to the recurrent
        weights.
    dnn_blocks : int
        The number of linear neural blocks to include.
    dnn_neurons : int
        The number of neurons in the linear layers.
    use_rnnp: bool
        If True, a linear projection layer is added between RNN layers.
    projection_dim : int
        The number of neurons in the projection layer.
        This layer is used to reduce the size of the flattened
        representation obtained after the CNN blocks.

    Example
    -------
    >>> inputs = torch.rand([10, 15, 60])
    >>> model = CRDNN(input_shape=inputs.shape)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 15, 512])
    """

    def __init__(
        self,
        output_size=1024,
        input_shape=None,
        activation=nn.LeakyReLU,
        dropout=0.15,
        cnn_blocks=2,
        cnn_channels=(128, 256),
        cnn_kernelsize=(3, 3),
        time_pooling=False,
        time_pooling_size=4,
        freq_pooling_size=2,
        rnn_class=LSTM,
        inter_layer_pooling_size=(2, 2),
        using_2d_pooling=False,
        rnn_layers=2,
        rnn_neurons=1024,
        rnn_bidirectional=False,
        rnn_re_init=True,
        dnn_blocks=2,
        dnn_neurons=1024,
        projection_dim=1024,
        use_rnnp=False,
    ):
        if output_size is None and input_shape is None:
            raise ValueError("Must specify one of input_size or input_shape")

        if input_shape is None:
            input_shape = [None, None, output_size]
        super().__init__(input_shape=input_shape)

        if cnn_blocks > 0:
            self.append(Sequential, layer_name="CNN")
        for block_index in range(cnn_blocks):
            self.CNN.append(
                CNN_Block,
                channels=cnn_channels[block_index],
                kernel_size=cnn_kernelsize,
                using_2d_pool=using_2d_pooling,
                pooling_size=inter_layer_pooling_size[block_index],
                activation=activation,
                dropout=dropout,
                layer_name=f"block_{block_index}",
            )

        if time_pooling:
            self.append(
                    Pooling1d(
                    pool_type="max",
                    input_dims=4,
                    kernel_size=time_pooling_size,
                    pool_axis=1,
                ),
                layer_name="time_pooling",
            )

        # This projection helps reducing the number of parameters
        # when using large number of CNN filters.
        # Large numbers of CNN filters + large features
        # often lead to very large flattened layers.
        # This layer projects it back to something reasonable.
        if projection_dim != -1:
            self.append(Sequential, layer_name="projection")
            self.projection.append(
                Linear,
                n_neurons=projection_dim,
                bias=True,
                combine_dims=True,
                layer_name="linear",
            )
            self.projection.append(
                LayerNorm, layer_name="norm"
            )
            self.projection.append(activation(), layer_name="act")

        if rnn_layers > 0:
            if use_rnnp:
                self.append(Sequential, layer_name="RNN")
                for _ in range(rnn_layers):
                    self.append(
                        rnn_class,
                        hidden_size=rnn_neurons,
                        num_layers=1,
                        bidirectional=rnn_bidirectional,
                        re_init=rnn_re_init,
                    )
                    self.append(
                        Linear,
                        n_neurons=dnn_neurons,
                        bias=True,
                        combine_dims=True,
                    )
                    self.append(nn.Dropout(p=dropout))
            else:
                self.append(
                    rnn_class,
                    layer_name="RNN",
                    hidden_size=rnn_neurons,
                    num_layers=rnn_layers,
                    dropout=dropout,
                    bidirectional=rnn_bidirectional,
                    re_init=rnn_re_init,
                )

        if dnn_blocks > 0:
            self.append(Sequential, layer_name="DNN")
        for block_index in range(dnn_blocks):
            self.DNN.append(
                DNN_Block,
                neurons=dnn_neurons,
                activation=activation,
                dropout=dropout,
                layer_name=f"block_{block_index}",
            )

class CNN_Block(Sequential):
    """CNN Block, based on VGG blocks.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    channels : int
        Number of convolutional channels for the block.
    kernel_size : tuple
        Size of the 2d convolutional kernel
    activation : torch.nn.Module class
        A class to be used for instantiating an activation layer.
    using_2d_pool : bool
        Whether to use 2d pooling or only 1d pooling.
    pooling_size : int
        Size of pooling kernel, duplicated for 2d pooling.
    dropout : float
        Rate to use for dropping channels.

    Example
    -------
    >>> inputs = torch.rand(10, 15, 60)
    >>> block = CNN_Block(input_shape=inputs.shape, channels=32)
    >>> outputs = block(inputs)
    >>> outputs.shape
    torch.Size([10, 15, 30, 32])
    """

    def __init__(
        self,
        input_shape,
        channels,
        kernel_size=[3, 3],
        activation=nn.LeakyReLU,
        using_2d_pool=False,
        pooling_size=2,
        dropout=0.15,
    ):
        super().__init__(input_shape=input_shape)
        self.append(
             Conv2d,
            out_channels=channels,
            kernel_size=kernel_size,
            layer_name="conv_1",
        )
        self.append(LayerNorm, layer_name="norm_1")
        self.append(activation(), layer_name="act_1")
        self.append(
            Conv2d,
            out_channels=channels,
            kernel_size=kernel_size,
            layer_name="conv_2",
        )
        self.append(LayerNorm, layer_name="norm_2")
        self.append(activation(), layer_name="act_2")

        if using_2d_pool:
            self.append(
                    Pooling2d(
                    pool_type="max",
                    kernel_size=(pooling_size, pooling_size),
                    pool_axis=(1, 2),
                ),
                layer_name="pooling",
            )
        else:
            self.append(
                    Pooling1d(
                    pool_type="max",
                    input_dims=4,
                    kernel_size=pooling_size,
                    pool_axis=2,
                ),
                layer_name="pooling",
            )

        self.append(
            Dropout2d(drop_rate=dropout), layer_name="drop"
        )


class DNN_Block(Sequential):
    """Block for linear layers.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    neurons : int
        Size of the linear layers.
    activation : torch.nn.Module class
        Class definition to use for constructing activation layers.
    dropout : float
        Rate to use for dropping neurons.

    Example
    -------
    >>> inputs = torch.rand(10, 15, 128)
    >>> block = DNN_Block(input_shape=inputs.shape, neurons=64)
    >>> outputs = block(inputs)
    >>> outputs.shape
    torch.Size([10, 15, 64])
    """

    def __init__(
        self, input_shape, neurons, activation=nn.LeakyReLU, dropout=0.15
    ):
        super().__init__(input_shape=input_shape)
        self.append(
            Linear, n_neurons=neurons, layer_name="linear",
        )
        self.append(BatchNorm1d, layer_name="norm")
        self.append(activation(), layer_name="act")
        self.append(nn.Dropout(p=dropout), layer_name="dropout")

class Attention(nn.Module):
    def __init__(
        self,
        output_size=1024,
        dropout = 0.01,
    ):
        super(Attention, self).__init__()
        self.output_size = output_size
        self.q = nn.Linear(output_size, output_size)
        self.k = nn.Linear(output_size, output_size)
        self.v = nn.Linear(output_size, output_size)
        self.out = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.LayerNorm(output_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
    
    def forward(self,q, k, v):
        q = self.q(q) # [4, 10, 1024] 
        k = self.k(k) # [4, 1000, 1024]
        v = self.v(v) # [4, 1000, 1024]

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.output_size)
        # clip
        score = torch.clip(score, 0) # >0 正相关保留
        score = F.softmax(score, dim=-1) # [4, 10, 1000]
        # print(q.shape, k.shape, v.shape, score.shape)
        x = torch.matmul(score, v) #  [4, 10, 1000] * [4, 1000, 1024] --> [4, 10, 1024]
        x = self.out(x)
        # print(x.shape)
        return x

class Decoder_layer(nn.Module):
    def __init__(
        self,
        output_size=1024,
        bidirectional=False,
        consider_as_one_hot=False,
        dropout=0.15,
        blank_id=0
    ):
        super(Decoder_layer, self).__init__() 
        self.rnn = nn.LSTM(input_size=output_size, hidden_size=output_size,num_layers=2, batch_first=True, dropout=0.15, bidirectional=bidirectional)
        self.l1 = nn.Sequential(
            nn.Linear(output_size, output_size, bias=True),
            nn.LayerNorm(output_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout)
        )
           
        self.att = Attention(output_size=output_size, dropout=dropout)
        self.ln = nn.LayerNorm(output_size)
    
    def forward(self, x, en_x):
        x, _ = self.rnn(x)
        x = self.l1(x)
        x = self.att(x, en_x, en_x)
        x = self.ln(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        num_classifies,
        consider_as_one_hot=False,
        embedding_dim=1024,
        output_size=1024,
        num_block=2,
        bidirectional=False,
        dropout=0.15,
        blank_id=0,
     ):
        super(Decoder, self).__init__()
        self.num_block = num_block
        self.embedding = Embedding(num_embeddings=num_classifies, embedding_dim=embedding_dim, consider_as_one_hot=False)
        self.l1 = nn.Sequential(
                nn.Linear(embedding_dim, output_size, bias=True),
                nn.LayerNorm(output_size),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout)
                )
        self.blocks = nn.ModuleList()
        for _ in range(self.num_block):
            self.blocks.append(
                Decoder_layer(
                    output_size=output_size,
                    bidirectional=False,
                    consider_as_one_hot=False,
                    dropout=0.15,
                    blank_id=0,
                )
            ) 
 
    def forward(self, x, x_en):
        x = self.embedding(x)
        x = self.l1(x) # (B, T, D)
        for layer in self.blocks:
            x = layer(x, x_en)
        return x


class Encoder_Decoer(nn.Module):
     
    def __init__(
        self,
        num_classifies,
        feat_shape=None,
        output_size=1024, 
        embedding_dim=512,
        dropout=0.15,
        de_num_layer=2,
    ):
        super(Encoder_Decoer, self).__init__()
        # self.output_size = output_size 
        self.encoder = Encoder(input_shape=feat_shape, output_size=output_size)
        self.decoder = Decoder(num_classifies, embedding_dim=embedding_dim, num_block=de_num_layer, output_size=output_size)
        self.en_out = nn.Linear(output_size, num_classifies, bias=False)
        self.de_out = nn.Linear(output_size, num_classifies, bias=False)     
         

    def decode(self, feats):
        en = self.encoder(feats)
        out = self.en_out(en)
        return en
 
    def forward(self, feats, targets_bos=None):
        x_en = self.encoder(feats) #  (B, T, D)
        en_out = self.en_out(x_en)
        
        x_de = self.decoder(targets_bos, x_en)
        de_out = self.de_out(x_de)
        return en_out, de_out


'''
torch.manual_seed(2021)
feats = torch.randn(4, 1500, 80).to("cuda:0")
targets = torch.randint(2, 4078,(4,20)).to("cuda:0")
print("input_feats.shape:{}".format(feats.shape))
print("input_targets.shape:{}".format(targets.shape))
net = Encoder_Decoer(4078, feat_shape=feats.shape).to("cuda:0")
print(net)
flops, params = profile(net, inputs=(feats, targets,))
print("flops:{}\tparams:{}".format(flops, params))
while 1:
    res = net(feats, targets)
    print(res[0].size(), res[1].size())
    torch.cuda.empty_cache()
'''
