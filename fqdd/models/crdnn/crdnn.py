from typing import List

import torch
import torch.nn as nn

from fqdd.decoders.search import DecodeResult
from fqdd.models.crdnn.decoder import CrdnnDecoder
from fqdd.models.crdnn.encoder import CrdnnEncoder
from fqdd.modules.CTC import CTC
from fqdd.modules.losses import LabelSmoothingLoss
from fqdd.text.tokenize_utils import add_sos_eos, remove_duplicates_and_blank
from fqdd.utils.common import th_accuracy
from fqdd.utils.mask import make_pad_mask


class CRDNN(nn.Module):

    def __init__(
            self,
            model_conf,

    ):
        '''
                    num_classifies,
            feat_shape=None,
            output_size=1024,
            embedding_dim=512,
            dropout=0.15,
            de_num_layer=2,
        Parameters
        ----------
        model_conf
        '''
        super(CRDNN, self).__init__()
        self.model_conf = model_conf

        encoder_conf = model_conf["encoder"]
        decoder_conf = model_conf["decoder"]
        use_cmvn = model_conf["use_cmvn"]
        cmvn_file = model_conf["cmvn_file"]

        self.vocab_size = model_conf["vocab_size"]
        self.special_tokens = model_conf["special_tokens"]
        self.lsm_weight = model_conf["lsm_weight"]
        self.ctc_weight = model_conf["ctc_weight"]
        self.encoder = CrdnnEncoder(encoder_conf, use_cmvn, cmvn_file)
        self.decoder = CrdnnDecoder(self.vocab_size, decoder_conf)
        ctc_conf = model_conf.get("ctc_conf", None)
        self.ctcloss = CTC(self.vocab_size, decoder_conf.get("encoder_output_size", 256),
                           blank_id=ctc_conf.get("ctc_blank_id") if "ctc_blank_id" in ctc_conf else 0)
        self.att_loss = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=self.lsm_weight,
            normalize_length=self.length_normalized_loss,
        )

    def forward(self, xs, xs_lens, padding_ys, ys_lens):
        encoder_out = self.encoder(xs)

        ctcloss, y_hats = self.ctcloss(encoder_out, xs_lens, padding_ys, ys_lens)

        ys_in_pad, ys_out_pad = add_sos_eos(padding_ys, self.sos, self.eos, self.ignore_id)

        decoder_out = self.decoder(ys_in_pad, encoder_out)

        loss_att = self.att_loss(decoder_out, ys_out_pad)
        loss = self.ctc_weight * ctcloss + (1 - self.ctc_weight) * loss_att

        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id
        )

        info_dicts = {
            "loss": loss,
            "ctc_loss": ctcloss,
            "att_loss": loss_att,
            "th_acc": acc_att,
            "encoder_out": encoder_out
        }

        return info_dicts

    def ctc_logprobs(self,
                     encoder_out: torch.Tensor,
                     blank_penalty: float = 0.0,
                     blank_id: int = 0):
        if blank_penalty > 0.0:
            logits = self.ctc.ctc_lo(encoder_out)
            logits[:, :, blank_id] -= blank_penalty
            ctc_probs = logits.log_softmax(dim=2)
        else:

            ctc_probs = self.ctcloss.log_softmax(encoder_out)

        return ctc_probs

    def ctc_greedy_search(
            self,
            ctc_probs: torch.Tensor,
            ctc_lens: torch.Tensor,
            blank_id: int = 0
    ) -> List[DecodeResult]:
        batch_size = ctc_probs.shape[0]
        maxlen = ctc_probs.size(1)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(ctc_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, blank_id)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)

        results = []
        for hyp in hyps:
            r = remove_duplicates_and_blank(hyp, blank_id)
            results.append(r)
        return results

    def decode(self,
               speech,
               speech_lengths,
               beam_size: int = 10,
               blank_id: int = 0,
               blank_penalty: float = 0.0,
               methods: List = ["ctc_greedy_search"]
               ):
        assert speech.shape[0] == speech_lengths.shape[0]
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc_logprobs(encoder_out, blank_penalty, blank_id)

        results = {}
        if 'ctc_greedy_search' in methods:
            results['ctc_greedy_search'] = self.ctc_greedy_search(
                ctc_probs, encoder_lens, blank_id)

        return results


torch.manual_seed(2021)
feats = torch.randn(4, 1500, 80).to("cpu")
targets = torch.randint(2, 4078, (4, 20)).to("cpu")
print("input_feats.shape:{}".format(feats.shape))
print("input_targets.shape:{}".format(targets.shape))
net = CRDNN().to("cpu")
print(net)
while 1:
    res = net(feats, targets)
    print(res[0].size(), res[1].size())
    torch.cuda.empty_cache()
