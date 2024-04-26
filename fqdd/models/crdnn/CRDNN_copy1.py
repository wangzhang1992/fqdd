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

class CRDNN_NET(Sequential):

    def __init__(self,
            input_shape=None,
            cnn_block_num=2,
            cnn_kernel_size=(3, 3),
            cnn_channels=(256, 512),
            inter_layer_pooling_size=(2, 2),
            using_2d_pooling=False,
            time_pooling=False,
            time_pooling_size=2,
            rnn_block_num=2,
            dropout=0.15,
            rnn_neurons=LiGRU,
            rnn_hidden_size=512,
            re_init=True,
            inter_rnn_num_layers=2,
            activation_layer='leaky_relu',
            bidirectional=False,
            dnn_neurons=512,
            inter_linear_layer_num=2,
            normalization='layernorm'):
            
        super().__init__(input_shape=input_shape)
        self.input_shape = input_shape
        
        if normalization == 'layernorm':
            layer_norm = LayerNorm
        if activation_layer == 'leaky_relu':
            activation = nn.LeakyReLU
        
        #self.append(Sequential, layer_name='batch_norm')
        #self.batch_norm.append((nn.BatchNorm1d(
        #    input_shape[1])),
        #    layer_name='batchnorma')

        self.append(Sequential, layer_name='CNN')
        for block_index in range(cnn_block_num):
            self.CNN.append(
                 CNN_Block,
                 channel=cnn_channels[block_index],
                 kernel_size=cnn_kernel_size,
                 using_2d_pool=using_2d_pooling,
                 pooling_size=inter_layer_pooling_size[block_index],
                 activation=activation,
                 dropout=dropout,
                 layer_name=f"block_{block_index}")

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

        self.append(Sequential, layer_name='projection')
        self.projection.append(
            Linear,
            n_neurons=rnn_hidden_size,
            bias=True,
            combine_dims=True,
            layer_name='linear')

        self.projection.append(
            layer_norm,
            layer_name='norm')
        self.projection.append((activation()), layer_name='act')
        self.projection.append(nn.Dropout(p=dropout), layer_name="project_drop")

        self.append(Sequential, layer_name='RNN')
        self.RNN.append(
            RNN_Block,
            hidden_size=rnn_hidden_size,
            rnn_neurons=rnn_neurons,
            re_init=re_init,
            num_layers=inter_rnn_num_layers,
            bidirectional=bidirectional,
            nonlinearity=activation_layer,
            dropout=dropout,
            normalization=normalization,
            layer_name=f"block_rnn")
        
        self.append(Sequential, layer_name='Line')
        self.Line.append(
            DNN_Block,
            n_neurons=dnn_neurons,
            num_layers=inter_linear_layer_num,
            dropout=dropout,
            layer_name="block_linear")
            

class CNN_Block(Sequential):

    def __init__(
            self,
            input_shape,
            channel,
            kernel_size=(3, 3),
            activation=nn.LeakyReLU,
            normalization=LayerNorm,
            using_2d_pool=False,
            pooling_size=2,
            dropout=0.15
        ):
        
        super().__init__(input_shape=input_shape)
        
        self.append(
            Conv2d,
            out_channels=channel,
            kernel_size=kernel_size,
            layer_name='conv_1')

        self.append(normalization, layer_name='norm_1')
        self.append(activation(), layer_name='act_1')
        self.append(
            Conv2d,
            out_channels=channel,
            kernel_size=kernel_size,
            layer_name='conv_2')

        self.append(normalization, layer_name='norm_2')
        self.append((activation()), layer_name='act_2')
        
        if using_2d_pool:
            self.append(
                Pooling2d(
                    pool_type='max',
                    kernel_size=(pooling_size, pooling_size),
                    pool_axis=(1, 2)
                ),
                layer_name='pooling')
        else:
            self.append(
                Pooling1d(
                    pool_type='max',
                    input_dims=4,
                    kernel_size=pooling_size,
                    pool_axis=2
                ),
                layer_name='pooling'
        )
        self.append(
            Dropout2d(drop_rate=dropout),
            layer_name='drop'
        )

'''         
inputs = torch.rand(4, 1500, 80)
model = CNN_Block(inputs.shape, 256, using_2d_pool=False)
print(model)
output = model(inputs)
print(output.shape)
'''

class RNN_Block(Sequential):

    def __init__(
            self,
            hidden_size,
            input_shape=None,
            rnn_neurons=LiGRU,
            num_layers=1,
            re_init=True,
            dropout=0.0,
            nonlinearity='leaky_relu',
            normalization='layernorm',
            bidirectional=False
        ):
        
        super().__init__(input_shape=input_shape)
        
        self.append(rnn_neurons,
            hidden_size=hidden_size,
            re_init=re_init,
            bidirectional=bidirectional,
            nonlinearity=nonlinearity,
            normalization=normalization,
            dropout=dropout,
            num_layers=num_layers,
            layer_name='liGRU_1')
        
        #self.append(LayerNorm, layer_name='norm_1')
        #self.append(nn.Dropout(p=dropout))
        #self.append((nn.LeakyReLU()), layer_name='act_1')


class DNN_Block(Sequential):

    def __init__(
            self,
            n_neurons,
            input_shape=None,
            num_layers=1,
            activation=nn.LeakyReLU,
            bias=True,
            dropout=0.15,
            combine_dims=False
        ):
        
        super().__init__(input_shape=input_shape)
        
        for block_index in range(num_layers):
            self.append(
                Linear,
                n_neurons=n_neurons,
                bias=bias,
                combine_dims=combine_dims,
                layer_name=f"linear_{block_index}")

            self.append(BatchNorm1d, layer_name=f"norm_{block_index}")
            self.append(activation(), layer_name=f"act_{block_index}")
            self.append(nn.Dropout(p=dropout), layer_name=f"dropout_{block_index}")

class Apply_CRDNN1(nn.Module):

    def __init__(
            self,
            n_neurons,
            input_shape=None,
            cnn_block_num=2,
            cnn_kernel_size=(3, 3),
            cnn_channels=(128, 256),
            inter_layer_pooling_size=(2, 2),
            using_2d_pooling=False,
            time_pooling=True,
            time_pooling_size=4,
            dropout=0.15,
            rnn_neurons=LiGRU,
            rnn_hidden_size=512,
            re_init=True,
            inter_rnn_num_layers=2,
            activation_layer='leaky_relu',
            bidirectional=True,
            dnn_neurons=1024,
            normalization='layernorm',
            inter_linear_layer_num=2
        ):
        super(Apply_CRDNN, self).__init__()
        self.crnn = CRDNN_NET(
            n_neurons,
            input_shape=input_shape,
            cnn_block_num=cnn_block_num,
            cnn_kernel_size=cnn_kernel_size,
            cnn_channels=cnn_channels,
            inter_layer_pooling_size=inter_layer_pooling_size,
            using_2d_pooling=using_2d_pooling,
            time_pooling=time_pooling,
            time_pooling_size=time_pooling_size,
            dropout=dropout,
            rnn_neurons=rnn_neurons,
            rnn_hidden_size=rnn_hidden_size,
            re_init=re_init,
            inter_rnn_num_layers=inter_rnn_num_layers,
            activation_layer=activation_layer,
            bidirectional=bidirectional,
            dnn_neurons=dnn_neurons,
            inter_linear_layer_num=inter_linear_layer_num,
            normalization=normalization
        )
        #self.batch_norm = nn.BatchNorm1d(input_shape[1])
    def forward(self, x):
        print(x.shape)
        #x = self.batch_norm(x)
        x = self.crnn(x.detach())
        return x

class CRDNN_layer(Sequential):
    def __init__(self, n_neurons, input_shape, layer_num=1):
        super().__init__(input_shape=input_shape)
        
        self.append(Sequential, layer_name='CRDNN')
        for block_index in range(layer_num):
            self.CRDNN.append(
                 CRDNN_NET,
                 layer_name=f"block_{block_index}")
        
        self.append(Sequential, layer_name="Decode")
        self.Decode.append(
            Linear,
            n_neurons=n_neurons,
            bias=False,
            layer_name="decode_line"
        )
         
class Apply_CRDNN(nn.Module):

    def __init__(self, n_neurons, input_shape, layer_num=1):
       super(Apply_CRDNN, self).__init__()         
       self.crdnn = CRDNN_layer(n_neurons, input_shape, layer_num)

    def forward(self, x):
        print(x.shape)
        #x = self.batch_norm(x)
        x = self.crdnn(x.detach())
        return x 

def test():
    # btch=16: flops:1048655872000.0       params:21151534.0
    cuda = "cuda:2"
    torch.manual_seed(2021)
    x_train = [torch.randn(4, 500, 120) for i in range(10)]
    inputd = torch.randn(4, 500, 120).to(cuda)
    print("input.shape:{}".format(inputd.shape))
    net = Apply_CRDNN(4078, input_shape=inputd.shape, layer_num=2).to(cuda)
    #for name, layer in net.named_parameters():
    #    print(name, layer)
    print(net)
    flops, params = profile(net, inputs=(inputd,))
    print("flops:{}\tparams:{}".format(flops, params))
    for batch_x in x_train:
        res = net(batch_x.to(cuda))
        print(res.size())
test()
