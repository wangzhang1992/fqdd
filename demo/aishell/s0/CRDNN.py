"""A combination of Convolutional, Recurrent, and Fully-connected networks.

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Ju-Chieh Chou 2020
 * Titouan Parcollet 2020
 * Abdel 2020
"""
import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import functools

sys.path.insert(0, "./")
from fqdd.nnets.containers import Sequential
from fqdd.nnets.CNN import Conv2d
from fqdd.nnets.dropout import Dropout2d
from fqdd.nnets.normalization import LayerNorm, BatchNorm1d
from fqdd.nnets.pooling import Pooling1d, Pooling2d
from fqdd.nnets.linear import Linear
from fqdd.nnets.embedding import Embedding
from fqdd.utils.init_tokenizer import add_sos_eos
from fqdd.nnets.base_utils import ACTIVATION_CLASSES, RNN_CLASSES


def length_to_mask(length, max_len=None, dtype=None, device=None):
    """Creates a binary mask for each sequence.

    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : tensor
        The binary mask.

    Example
    -------
    >>> length=torch.Tensor([1,2,3])
    >>> mask=length_to_mask(length)
    >>> mask
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


def truncate(predictions, targets, allowed_len_diff=3):
    """Ensure that predictions and targets are the same length.

    Arguments
    ---------
    predictions : torch.Tensor
        First tensor for checking length.
    targets : torch.Tensor
        Second tensor for checking length.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    """
    len_diff = predictions.shape[1] - targets.shape[1]
    if len_diff == 0:
        return predictions, targets
    elif abs(len_diff) > allowed_len_diff:
        raise ValueError(
            "Predictions and targets should be same length, but got %s and "
            "%s respectively." % (predictions.shape[1], targets.shape[1])
        )
    elif len_diff < 0:
        return predictions, targets[:, : predictions.shape[1]]
    else:
        return predictions[:, : targets.shape[1]], targets


def calculate_loss(log_probs, targets, input_lengths, target_lengths):
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

    batch = log_probs.size(0)
    log_probs = F.log_softmax(log_probs, dim=2)
    log_probs = log_probs.detach().transpose(0, 1)  # T x B x D
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean')
    loss = ctc_loss(log_probs.to("cpu"), targets.to("cpu"), input_lengths.to("cpu"), target_lengths.to("cpu"))
    loss = loss / batch
    return loss


def nll_loss(
        log_probabilities,
        targets,
        length=None,
        label_smoothing=0.0,
        allowed_len_diff=3,
        reduction="mean",
):
    """Computes negative log likelihood loss.

    Arguments
    ---------
    log_probabilities : torch.Tensor
        The probabilities after log has been applied.
        Format is [batch, log_p] or [batch, frames, log_p].
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames].
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    >>> nll_loss(torch.log(probs), torch.tensor([1, 1]))
    tensor(1.2040)
    """

    def compute_masked_loss(
            loss_fn,
            predictions,
            targets,
            length=None,
            label_smoothing=0.0,
            reduction="mean",
    ):
        """Compute the true average loss of a set of waveforms of unequal length.

        Arguments
        ---------
        loss_fn : function
            A function for computing the loss taking just predictions and targets.
            Should return all the losses, not a reduction (e.g. reduction="none").
        predictions : torch.Tensor
            First argument to loss function.
        targets : torch.Tensor
            Second argument to loss function.
        length : torch.Tensor
            Length of each utterance to compute mask. If None, global average is
            computed and returned.
        label_smoothing: float
            The proportion of label smoothing. Should only be used for NLL loss.
            Ref: Regularizing Neural Networks by Penalizing Confident Output
            Distributions. https://arxiv.org/abs/1701.06548
        reduction : str
            One of 'mean', 'batch', 'batchmean', 'none' where 'mean' returns a
            single value and 'batch' returns one per item in the batch and
            'batchmean' is sum / batch_size and 'none' returns all.
        """
        mask = torch.ones_like(targets)
        if length is not None:
            length_mask = length_to_mask(
                length * targets.shape[1], max_len=targets.shape[1],
            )

            # Handle any dimensionality of input
            while len(length_mask.shape) < len(mask.shape):
                length_mask = length_mask.unsqueeze(-1)
            length_mask = length_mask.type(mask.dtype)
            mask *= length_mask

        # Compute, then reduce loss
        loss = loss_fn(predictions, targets) * mask
        N = loss.size(0)
        if reduction == "mean":
            loss = loss.sum() / torch.sum(mask)
        elif reduction == "batchmean":
            loss = loss.sum() / N
        elif reduction == "batch":
            loss = loss.reshape(N, -1).sum(1) / mask.reshape(N, -1).sum(1)

        if label_smoothing == 0:
            return loss
        else:
            loss_reg = torch.mean(predictions, dim=1) * mask
            if reduction == "mean":
                loss_reg = torch.sum(loss_reg) / torch.sum(mask)
            elif reduction == "batchmean":
                loss_reg = torch.sum(loss_reg) / targets.shape[0]
            elif reduction == "batch":
                loss_reg = loss_reg.sum(1) / mask.sum(1)

            return -label_smoothing * loss_reg + (1 - label_smoothing) * loss

    log_probabilities = F.log_softmax(log_probabilities, dim=-1)

    if len(log_probabilities.shape) == 3:
        log_probabilities, targets = truncate(
            log_probabilities, targets, allowed_len_diff
        )
        log_probabilities = log_probabilities.transpose(1, -1)

    # Pass the loss function but apply reduction="none" first
    loss = functools.partial(torch.nn.functional.nll_loss, reduction="none")
    return compute_masked_loss(
        loss,
        log_probabilities,
        targets.long(),
        length,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


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
            encoder_conf
    ):
        output_size = encoder_conf.get("output_size", 1024)
        activation_type = encoder_conf.get("activation_type", "LeakyReLU")
        dropout = encoder_conf.get("dropout", 0.1)
        cnn_blocks = encoder_conf.get("cnn_blocks", 2)
        cnn_channels = encoder_conf.get("cnn_channels", (128, 256))
        cnn_kernelsize = encoder_conf.get("cnn_kernelsize", (3, 3))
        time_pooling = encoder_conf.get("time_pooling", False)
        time_pooling_size = encoder_conf.get("time_pooling_size", 4)
        rnn_class_type = encoder_conf.get("rnn_class_type", "lstm")
        rnn_layers = encoder_conf.get("rnn_layers", 2)
        rnn_neurons = encoder_conf.get("rnn_neurons", 1024)
        rnn_bidirectional = encoder_conf.get("rnn_bidirectional", False)
        rnn_re_init = encoder_conf.get("rnn_re_init", True)
        inter_layer_pooling_size = encoder_conf.get("inter_layer_pooling_size", (2, 2))
        using_2d_pooling = encoder_conf.get("using_2d_pooling", False)
        dnn_blocks = encoder_conf.get("dnn_blocks", 2, )
        dnn_neurons = encoder_conf.get("dnn_neurons", 1024)
        projection_dim = encoder_conf.get("projection_dim", 1024)
        use_rnnp = encoder_conf.get("use_rnnp", False)
        input_shape = None

        if output_size is None and input_shape is None:
            raise ValueError("Must specify one of input_size or input_shape")

        if input_shape is None:
            input_shape = [None, None, output_size]
        super().__init__(input_shape=input_shape)

        activation = ACTIVATION_CLASSES[activation_type]()
        rnn_class = RNN_CLASSES["rnn_class_type"]

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
            dropout=0.01,
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

    def forward(self, q, k, v):
        q = self.q(q)  # [4, 10, 1024]
        k = self.k(k)  # [4, 1000, 1024]
        v = self.v(v)  # [4, 1000, 1024]

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.output_size)
        # clip
        score = torch.clip(score, 0)  # >0 正相关保留
        score = F.softmax(score, dim=-1)  # [4, 10, 1000]
        # print(q.shape, k.shape, v.shape, score.shape)
        x = torch.matmul(score, v)  # [4, 10, 1000] * [4, 1000, 1024] --> [4, 10, 1024]
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
        self.rnn = nn.LSTM(input_size=output_size, hidden_size=output_size, num_layers=2, batch_first=True,
                           dropout=0.15, bidirectional=bidirectional)
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
            decoder_conf
    ):
        super(Decoder, self).__init__()

        vocab_size = decoder_conf.get("vocab_size")
        self.num_block = decoder_conf.get("num_block", 2)
        output_size = decoder_conf.get("output_size", 1024)
        drop_rate = decoder_conf.get("dropout", 0.1)
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=output_size,
                                   consider_as_one_hot=False)
        self.l1 = nn.Sequential(
            nn.Linear(output_size, output_size, bias=True),
            nn.LayerNorm(output_size),
            nn.LeakyReLU(),
            nn.Dropout(p=drop_rate)
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
        x = self.l1(x)  # (B, T, D)
        for layer in self.blocks:
            x = layer(x, x_en)
        return x


class Encoder_Decoer(nn.Module):

    def __init__(
            self,
            model_conf
    ):
        super(Encoder_Decoer, self).__init__()

        self.model_conf = model_conf
        encoder_conf = model_conf["encoder"]
        decoder_conf = model_conf["decoder"]
        use_cmvn = model_conf["use_cmvn"]
        cmvn_file = model_conf["cmvn_file"]

        self.vocab_size = model_conf["vocab_size"]
        self.special_tokens = model_conf["special_tokens"]
        self.lsm_weight = model_conf["lsm_weight"]
        self.ctc_weight = model_conf["ctc_weight"]

        self.sos = (self.vocab_size - 1 if self.special_tokens is None else
                    self.special_tokens.get("<sos>", self.vocab_size - 1))
        self.eos = (self.vocab_size - 1 if self.special_tokens is None else
                    self.special_tokens.get("<eos>", self.vocab_size - 1))
        self.ignore_id = -1

        self.encoder = Encoder(encoder_conf)
        self.decoder = Decoder(decoder_conf, self.vocab_size, embedding_dim=embedding_dim, num_block=de_num_layer,
                               output_size=output_size)
        self.en_out = nn.Linear(output_size, self.vocab_size, bias=False)
        self.de_out = nn.Linear(output_size, self.vocab_size, bias=False)

    def decode(self, feats):
        en = self.encoder(feats)
        out = self.en_out(en)
        return en

    def forward(self, feats, wav_lengths, ys, ys_lens):
        x_en = self.encoder(feats)  # (B, T, D)
        en_out = self.en_out(x_en)

        ys_in_pad, ys_out_pad = add_sos_eos(ys, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_lens + 1

        x_de = self.decoder(ys_in_pad, x_en)
        de_out = self.de_out(x_de)

        ctc_loss = calculate_loss(en_out, ys, wav_lengths, ys_lens)
        att_loss = nll_loss(de_out, ys_out_pad, ys_in_lens)

        loss = self.ctc_weight * ctc_loss + att_loss * (1 - self.ctc_weight)
        info_dicts = {
            "loss": loss,
            "ctc_loss": ctc_loss,
            "att_loss": att_loss,
            "encoder_out": en_out
        }
        return info_dicts


model_conf = {
    "encoder_conf": {
        "output_size": 1024,
        "activation": "LeakyReLU",
        "dropout": 0.1,
        "cnn_blocks": 2,
        "cnn_channels": (128, 256),
        "cnn_kernelsize": (3, 3),
        "time_pooling": False,
        "time_pooling_size": 4,
        "rnn_class": "lstm",
        "rnn_layers": 2,
        "rnn_neurons": 1024,
        "rnn_bidirectional": False,
        "rnn_re_init": True,
        "inter_layer_pooling_size": (2, 2),
        "using_2d_pooling": False,
        "dnn_blocks": 2,
        "dnn_neurons": 1024,
        "projection_dim": 1024,
        "use_rnnp": False,
    },

    "decoder_conf": {
        "vocab_size": 4078,
        "output_size": 1024,
        "num_block": 2,
        "dropout": 0.1
    }
}

torch.manual_seed(2024)
feats = torch.randn(4, 800, 80).to("cuda:0")
feat_length = torch.tensor([321, 134, 512, 800], dtype=torch.int32)
targets = torch.randint(2, 4078, (4, 20)).to("cuda:0")
targets_length = torch.tensor([9, 14, 20, 11], dtype=torch.int32)
print("input_feats.shape:{}".format(feats.shape))
print("input_targets.shape:{}".format(targets.shape))
net = Encoder_Decoer(model_conf)
print(net)
while 1:
    info_dicts = net(feats, feat_length, targets, targets_length)
    print(info_dicts)
