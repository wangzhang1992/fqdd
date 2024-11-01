import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


class Encoder(nn.Module):
    r"""Applies a multi-layer LSTM to an variable length input sequence.
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout=0.0, bidirectional=True, rnn_type='lstm'):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.dropout = dropout
        if self.bidirectional:
            dnn_dim = self.hidden_size * 2
        else:
            dnn_dim = self.hidden_size
       
        self.dnn1 = nn.Sequential(
                nn.Linear(input_size, hidden_size//2), 
                nn.Linear(hidden_size//2, hidden_size//2),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(hidden_size//2, hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=self.dropout)
        )

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=bidirectional)

        self.dnn2 = nn.Sequential(
                nn.Linear(dnn_dim, hidden_size), 
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(hidden_size, dnn_dim),
                nn.Linear(dnn_dim, dnn_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout)
        )

    def forward(self, padded_input):
        """
        Args:
            padded_input: B x T x D
            input_lengths: T
        Returns: output, hidden
            - **output**: B x T x H
            - **hidden**: (num_layers * num_directions) x T x H
        """
        rnn_input = self.dnn1(padded_input)
        rnn_output, hidden = self.rnn(rnn_input)
        output = self.dnn2(rnn_output)
        
        return output

    def flatten_parameters(self):
        self.rnn.flatten_parameters()


class DotProductAttention(nn.Module):
    r"""Dot product attention.
    Given a set of vector values, and a vector query, attention is a technique
    to compute a weighted sum of the values, dependent on the query.
    NOTE: Here we use the terminology in Stanford cs224n-2018-lecture11.
    """

    def __init__(self):
        super(DotProductAttention, self).__init__()
        # TODO: move this out of this class?
        # self.linear_out = nn.Linear(dim*2, dim)

    def forward(self, queries, values):
        """
        Args:
            queries: N x To x H
            values : N x Ti x H
        Returns:
            output: N x To x H
            attention_distribution: N x To x Ti
        """
        batch_size = queries.size(0)
        input_lengths = values.size(1)
        # (N, To, H) * (N, H, Ti) -> (N, To, Ti)
        attention_scores = torch.bmm(queries, values.transpose(1, 2))
        attention_distribution = F.softmax(
            attention_scores.view(-1, input_lengths), dim=1).view(batch_size, -1, input_lengths)
        # (N, To, Ti) * (N, Ti, H) -> (N, To, H)
        attention_output = torch.bmm(attention_distribution, values)

        return attention_output, attention_distribution


class Decoder(nn.Module):
    """
    """

    def __init__(self, vocab_size, embedding_dim=512, hidden_size=256,
                 num_layers=1, bidirectional_encoder=True):
        super(Decoder, self).__init__()
        # Hyper parameters
        # embedding + output
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # rnn
        self.bidirectional_encoder = bidirectional_encoder  # useless now
        self.hidden_size = hidden_size * 2 if self.bidirectional_encoder else hidden_size
        self.num_layers = num_layers

        self.encoder_hidden_size = self.hidden_size  # must be equal now
        # Components
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.ModuleList()
        self.rnn += [nn.LSTMCell(self.embedding_dim +
                                 self.encoder_hidden_size, self.hidden_size)]
        for l in range(1, self.num_layers):
            self.rnn += [nn.LSTMCell(self.hidden_size, self.hidden_size)]
        self.attention = DotProductAttention()
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_hidden_size + self.hidden_size,
                      self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.vocab_size))

    def zero_state(self, encoder_padded_outputs, H=None):
        N = encoder_padded_outputs.size(0)
        H = self.hidden_size if H == None else H
        return encoder_padded_outputs.new_zeros(N, H)

    def forward(self, padded_input, encoder_padded_outputs):
        """
        Args:
            padded_input: N x To
            # encoder_hidden: (num_layers * num_directions) x N x H
            encoder_padded_outputs: N x Ti x H
        Returns:
        """
        batch_size = padded_input.size(0)
        output_length = padded_input.size(1)
        # max_length = ys_in_pad.size(1) - 1  # TODO: should minus 1(sos)?

        # *********Init decoder rnn
        h_list = [self.zero_state(encoder_padded_outputs)]  # N * H
        c_list = [self.zero_state(encoder_padded_outputs)]  # N * H
        for l in range(1, self.num_layers):
            h_list.append(self.zero_state(encoder_padded_outputs))
            c_list.append(self.zero_state(encoder_padded_outputs))
        att_c = self.zero_state(encoder_padded_outputs,
                                H=encoder_padded_outputs.size(2))
        y_all = []

        # **********LAS: 1. decoder rnn 2. attention 3. concate and MLP
        embedded = self.embedding(padded_input)  # B * N * embedding_dim

        for t in range(output_length):
            # step 1. decoder RNN: s_i = RNN(s_i−1,y_i−1,c_i−1)
            rnn_input = torch.cat((embedded[:, t, :], att_c), dim=1)
            h_list[0], c_list[0] = self.rnn[0](
                rnn_input, (h_list[0], c_list[0]))  # N * H
            for l in range(1, self.num_layers):
                h_list[l], c_list[l] = self.rnn[l](
                    h_list[l - 1], (h_list[l], c_list[l]))
            rnn_output = h_list[-1]  # below unsqueeze: (N x H) -> (N x 1 x H)
            # step 2. attention: c_i = AttentionContext(s_i,h)
            att_c, att_w = self.attention(rnn_output.unsqueeze(dim=1),
                                          encoder_padded_outputs)
            att_c = att_c.squeeze(dim=1)
            # step 3. concate s_i and c_i, and input to MLP
            mlp_input = torch.cat((rnn_output, att_c), dim=1)
            predicted_y_t = self.mlp(mlp_input)
            y_all.append(predicted_y_t)

        out = torch.stack(y_all, dim=1)  # N x To x C
        # **********Cross Entropy Loss
        # F.cross_entropy = NLL(log_softmax(input), target))
        out = out.view(batch_size * output_length, self.vocab_size)

        return out


class LAS(nn.Module):
    """Sequence-to-Sequence architecture with configurable encoder and decoder.
    """

    def __init__(self, input_dim, vocab_size, params):
        super(LAS, self).__init__()
        self.paras = params
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.encoder, self.decoder = self.init_encoder_decoder()
        self.en_l = nn.Linear(self.paras.hiddle_size * 2 if self.paras.bidirectional else self.paras.hiddle_size, vocab_size, bias=False)

    def init_encoder_decoder(self):
        encoder = Encoder(self.input_dim, self.paras.hiddle_size, self.paras.e_num_layers,
                          dropout=self.paras.drop, bidirectional=self.paras.bidirectional,
                          rnn_type="lstm")
        decoder = Decoder(self.vocab_size, self.paras.embedding_dim, self.paras.hiddle_size, self.paras.d_num_layers,
                          bidirectional_encoder=self.paras.bidirectional)
        return encoder, decoder

    def forward(self, padded_input, padded_target=None):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        decode_output = None
        #encoder_padded_outputs, _ = self.encoder(padded_input)
        encoder_padded_outputs = self.encoder(padded_input)
        if padded_target is not None:
            decode_output = self.decoder(padded_target, encoder_padded_outputs)
        encoder_padded_outputs = self.en_l(encoder_padded_outputs)
        return encoder_padded_outputs, decode_output


def calculate_celoss(preds, padded_output, ignore_index=0, reduction="mean"):
    ce_loss = F.cross_entropy(preds, padded_output.view(-1),
                              ignore_index=ignore_index,
                              reduction=reduction)
    return ce_loss

'''
train_x = torch.randn(4, 100, 80)
x_len = torch.LongTensor([49, 79, 80, 100])
# unk:0, bos:1, eos: 2
train_y = [[9, 3, 5, 8, 6], [4, 7, 6, 5], [6, 8, 9, 4], [7, 6, 3]]
train_bos = torch.LongTensor([[1, 9, 3, 5, 8, 6], [1, 4, 7, 6, 5, 0], [1, 6, 8, 9, 4, 0], [1, 7, 6, 3, 0, 0]])
train_eos = torch.LongTensor([[9, 3, 5, 8, 6, 2], [4, 7, 6, 5, 2, 0], [6, 8, 9, 4, 2, 0], [7, 6, 3, 2, 0, 0]])

params = argparse.ArgumentParser(description="model config", add_help=True)
params.add_argument("--hiddle_size", type=int, default=256)
params.add_argument("--e_num_layers", type=int, default=3)
params.add_argument("--d_num_layers", type=int, default=1)
params.add_argument("--drop", type=float, default=0.2)
params.add_argument("--bidirectional", type=bool, default=True)
params.add_argument("--etype", type=str, default="lstm")
params.add_argument("--embedding_dim", type=int, default=512)

vocab_size = 512
input_dim = train_x.shape[-1]
las_model = LAS(input_dim, vocab_size, params.parse_args())
en, de = las_model(train_x, train_bos)
ce_loss = calculate_celoss(de, train_eos, ignore_index=0, reduction="mean")
print(f"res={ce_loss}")
'''
