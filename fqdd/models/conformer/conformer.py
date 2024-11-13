import torch
import torch.nn as nn

from typing import List
from fqdd.models.CTC import CTC
from fqdd.models.conformer.decoder import TransformerDecoder
from fqdd.models.conformer.encoder import ConformerEncoder
from fqdd.models.search import DecodeResult
from fqdd.nnets.losses import LabelSmoothingLoss
from fqdd.text.tokenize_utils import add_sos_eos, reverse_pad_list, remove_duplicates_and_blank
from fqdd.utils.common import IGNORE_ID, th_accuracy
from fqdd.utils.mask import make_pad_mask


class Conformer(nn.Module):
    def __init__(
            self,
            model_conf,
    ):
        super(Conformer, self).__init__()

        self.model_conf = model_conf

        encoder_conf = model_conf["encoder"]
        decoder_conf = model_conf["decoder"]
        use_cmvn = model_conf["use_cmvn"]
        cmvn_file = model_conf["cmvn_file"]

        self.vocab_size = model_conf["vocab_size"]
        self.special_tokens = model_conf["special_tokens"]
        self.ignore_id = IGNORE_ID
        self.lsm_weight = model_conf["lsm_weight"]
        self.ctc_weight = model_conf["ctc_weight"]

        self.reverse_weight = 0.0

        self.length_normalized_loss = model_conf["length_normalized_loss"]
        self.sos = (self.vocab_size - 1 if self.special_tokens is None else
                    self.special_tokens.get("<sos>", self.vocab_size - 1))
        self.eos = (self.vocab_size - 1 if self.special_tokens is None else
                    self.special_tokens.get("<eos>", self.vocab_size - 1))
        # print(self.sos, self.eos)

        self.encoder = ConformerEncoder(encoder_conf, use_cmvn, cmvn_file)
        self.decoder = TransformerDecoder(self.vocab_size, decoder_conf)

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

        encoder_out, encoder_mask = self.encoder(xs, xs_lens)
        print("encoder_out.shape:{}\nencoder_out[0][0]{}".format(encoder_out.shape, encoder_out[0][0]))

        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        print("encoder_out_lens:{}".format(encoder_out_lens))

        ctcloss, y_hats = self.ctcloss(encoder_out, encoder_out_lens, padding_ys, ys_lens)
        print("y_hats.shape:{}\ny_hats[0][0]{}".format(y_hats.shape, y_hats[0][0]))

        ys_in_pad, ys_out_pad = add_sos_eos(padding_ys, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_lens + 1

        r_ys_pad = reverse_pad_list(padding_ys, ys_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.ignore_id)

        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask, ys_in_pad, ys_in_lens, r_ys_in_pad,
                                                     self.reverse_weight)
        print("decoder_out.shape:{}\nr_decoder_out:{}".format(decoder_out.shape, r_decoder_out))

        loss_att = self.att_loss(decoder_out, ys_out_pad)
        loss = self.ctc_weight * ctcloss + (1 - self.ctc_weight) * loss_att
        # print(decoder_out.shape)
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
            # print("ctc_logprobs: encoder_out.shape:{}".format(encoder_out.shape))
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
        # print("ctc_greed: hyp:{}".format(hyps))
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
        # print("**********{}".format(ctc_probs.shape))
        results = {}
        if 'ctc_greedy_search' in methods:
            results['ctc_greedy_search'] = self.ctc_greedy_search(
                ctc_probs, encoder_lens, blank_id)

        return results
