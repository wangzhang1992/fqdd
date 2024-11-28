import torch
import torch.nn as nn
from fqdd.nnets.containers import Sequential
from fqdd.nnets.CNN import Conv2d
from fqdd.nnets.dropout import Dropout2d
from fqdd.nnets.normalization import LayerNorm, BatchNorm1d
from fqdd.nnets.pooling import Pooling1d, Pooling2d
from fqdd.nnets.linear import Linear


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
