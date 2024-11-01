import torch
import torch.nn as nn
from script.tools.argument import parse_arguments


class RNN_Model(nn.Module):
    def __init__(self, vocab_size, args):
        super(RNN_Model, self).__init__()
        self.vocab_size = vocab_size
        self.hiddle = args.hiddle_size
        self.layers = args.layer_size
        self.batch = args.batch_size
        self.drop = args.drop
        self.bi = args.bidirectional

        self.emb = nn.Embedding(self.vocab_size, args.emb_nodes)
        self.emdrop = nn.Dropout(p=0.3)
        self.rnn = nn.RNN(input_size=args.emb_nodes, hidden_size=args.hiddle_size, num_layers=self.layers,
                          batch_first=True, dropout=self.drop, bidirectional=self.bi)

        # self.blstm = nn.LSTM(self.input_dim, self.hiddle, self.layers, batch_first=True, dropout=self.drop,
        #                     bidirectional=self.bi)
        self.normal = nn.BatchNorm1d(args.max_label_length, affine=True)
        self.l1_dim = self.hiddle
        if self.bi:
            self.l1_dim = self.hiddle * 2
        self.line1 = nn.Sequential(nn.Linear(self.l1_dim, self.vocab_size))

    def forward(self, x):
        x = self.emdrop(self.emb(x))
        x, _ = self.rnn(x)
        x = self.normal(x)
        x = self.line1(x)
        return x


def test():
    args = parse_arguments()
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    ce_loss = torch.nn.CrossEntropyLoss()
    data = [torch.randint(1, 1246, (16, 10)) for i in range(20)]

    rnnm = RNN_Model(1246, args)
    # rnnm.to("cuda:1")
    optimize = torch.optim.Adam(rnnm.parameters(), lr=0.001)
    for i in range(20):
        for x in data:
            pre = rnnm(x)
            pre = torch.transpose(pre, 1, 2)
            optimize.zero_grad()
            loss = ce_loss(pre, x)
            loss.backward()
            optimize.step()
            print(loss.item())


'''
if __name__ == "__main__":
    test()
'''
