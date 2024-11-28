import torch
import torch.nn as nn

from fqdd.modules.model_utils import LayerDropModuleList
from fqdd.nnets.base_utils import FQDD_ACTIVATIONS
from fqdd.nnets.linear import Linear
from fqdd.nnets.normalization import LayerNorm
from fqdd.nnets.pooling import Pooling1d
from fqdd.models.crdnn.encoder_layer import CNN_Block, DNN_Block
from fqdd.utils.common import load_json_cmvn, GlobalCMVN


class CrdnnEncoder(nn.Module):
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
    """

    def __init__(
            self,
            encoder_conf,
            use_cmvn: bool = False,
            cmvn_file: str = None
    ):
        super(CrdnnEncoder, self).__init__()
        input_size = encoder_conf.get("input_size")
        output_size = encoder_conf.get("output_size", 256)
        activation_type = encoder_conf.get("activation_type", "swish")
        dropout_rate = encoder_conf.get("dropout_rate", 0.1)
        cnn_blocks = encoder_conf.get("num_blocks", 2)
        cnn_channels = encoder_conf.get("cnn_channels", (128, 256))
        cnn_kernel_size = encoder_conf.get("cnn_kernel_size", (3, 3))
        time_pooling = encoder_conf.get("time_pooling", False)
        time_pooling_size = encoder_conf.get("time_pooling_size", 4)
        freq_pooling_size = encoder_conf.get("freq_pooling_size", 2)
        stochastic_depth_rate = encoder_conf.get("stochastic_depth_rate", 0.0)
        rnn_class = encoder_conf.get("rnn_class", "lstm")
        inter_layer_pooling_size = encoder_conf.get("cnn_kernel_size", (2, 2))
        using_2d_pooling = encoder_conf.get("using_2d_pooling", False)
        rnn_layers = encoder_conf.get("rnn_layers", 2)
        rnn_neurons = encoder_conf.get("rnn_neurons", 1024)
        rnn_bidirectional = encoder_conf.get("rnn_bidirectional", False)
        rnn_re_init = encoder_conf.get("rnn_re_init", True)
        dnn_blocks = encoder_conf.get("dnn_blocks", 2)
        dnn_neurons = encoder_conf.get("dnn_neurons", 1024)
        projection_dim = encoder_conf.get("projection_dim", 1024)
        use_rnnp = encoder_conf.get("use_rnnp", False)

        if use_cmvn:
            mean, std = load_json_cmvn(cmvn_file)
            mean = torch.from_numpy(mean).float()
            std = torch.from_numpy(std).float()
            self.global_cmvn = GlobalCMVN(mean, std)

        self.cnn_block = LayerDropModuleList(p=stochastic_depth_rate, modules=[
            CNN_Block(
                channels=cnn_channels[block_index],
                kernel_size=cnn_kernel_size,
                using_2d_pool=using_2d_pooling,
                pooling_size=inter_layer_pooling_size[block_index],
                activation=FQDD_ACTIVATIONS[activation_type](),
                dropout=dropout_rate,
                layer_name=f"block_{block_index}",
            ) for block_index in range(cnn_blocks)
        ])



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
            self.projection.append(FQDD_ACTIVATIONS[activation_type](), layer_name="act")

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
                    self.append(nn.Dropout(p=dropout_rate))
            else:
                self.append(
                    rnn_class,
                    layer_name="RNN",
                    hidden_size=rnn_neurons,
                    num_layers=rnn_layers,
                    dropout=dropout_rate,
                    bidirectional=rnn_bidirectional,
                    re_init=rnn_re_init,
                )

        if dnn_blocks > 0:
            self.append(Sequential, layer_name="DNN")
        for block_index in range(dnn_blocks):
            self.DNN.append(
                DNN_Block,
                neurons=dnn_neurons,
                activation=FQDD_ACTIVATIONS[activation_type](),
                dropout=dropout_rate,
                layer_name=f"block_{block_index}",
            )
