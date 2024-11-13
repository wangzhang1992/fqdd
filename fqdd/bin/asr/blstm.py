import torch.nn as nn


class BlstmNet(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(BlstmNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hiddle = args.hiddle_size
        self.layers = args.layer_size
        self.batch = args.batch_size
        self.drop = args.drop
        self.bi = args.bidirectional
        self.blstm = nn.LSTM(self.input_dim, self.hiddle, self.layers, batch_first=True, dropout=self.drop,
                             bidirectional=self.bi)
        self.normal = nn.BatchNorm1d(self.batch, affine=True)
        self.l1_dim = self.hiddle
        if self.bi:
            self.l1_dim = self.hiddle * 2
        self.line1 = nn.Sequential(nn.Linear(self.l1_dim, self.output_dim))

    def forward(self, x):
        x, _ = self.blstm(x)
        x = self.normal(x.transpose(0, 1))
        x = self.line1(x.transpose(0, 1))
        # x = self.line1(x)
        return x

# batch=16: flops:904868659200.0	params:25675790.0
# torch.manual_seed(2021)
# args = parse_arguments()
# x_train = [torch.randn(16, 2200, 120) for i in range(10)]
# args.batch_size=16
# net = BlstmNet(120, 4078, args)
# input = torch.randn(16, 2200, 120)
# flops, params = profile(net, inputs=(input, ))
# print("flops:{}\tparams:{}".format(flops, params))
# for batch_x in x_train:
#     res = net(batch_x)
#     print(res.size())
