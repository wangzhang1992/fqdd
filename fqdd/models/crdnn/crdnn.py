import torch
import torch.nn as nn

from fqdd.models.crdnn.decoder import Decoder
from fqdd.models.crdnn.encoder import Encoder


class CRDNN(nn.Module):

    def __init__(
            self,
            num_classifies,
            feat_shape=None,
            output_size=1024,
            embedding_dim=512,
            dropout=0.15,
            de_num_layer=2,
    ):
        super(CRDNN, self).__init__()
        # self.output_size = output_size 
        self.encoder = Encoder(input_shape=feat_shape, output_size=output_size)
        self.decoder = Decoder(num_classifies, embedding_dim=embedding_dim, num_block=de_num_layer,
                               output_size=output_size)
        self.en_out = nn.Linear(output_size, num_classifies, bias=False)
        self.de_out = nn.Linear(output_size, num_classifies, bias=False)

    def decode(self, feats):
        en = self.encoder(feats)
        return en

    def forward(self, feats, targets_bos=None):
        x_en = self.encoder(feats)  # (B, T, D)
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
