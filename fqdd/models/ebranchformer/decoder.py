from typing import Tuple

import torch

from fqdd.models.ebranchformer.decoder_layer import DecoderLayer
from fqdd.modules.model_utils import FQDD_EMBEDDINGS, FQDD_MLPS, FQDD_ATTENTIONS
from fqdd.nnets.base_utils import FQDD_ACTIVATIONS, FQDD_NORMALIZES
from fqdd.utils.mask import make_pad_mask, subsequent_mask


class TransformerDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        src_attention: if false, encoder-decoder cross attention is not
                       applied, such as CIF model
        query_bias: whether use bias in attention.linear_q
        key_bias: whether use bias in attention.linear_k, False for whisper models.
        value_bias: whether use bias in attention.linear_v
        gradient_checkpointing: rerunning a forward-pass segment for each
            checkpointed segment during backward.
        tie_word_embedding: Tie or clone module weights depending of whether we are
            using TorchScript or not
    """

    def __init__(
            self,
            vocab_size,
            decoder_conf
    ):
        super().__init__()

        encoder_output_size = decoder_conf.get("encoder_output_size", 256)
        linear_units = decoder_conf.get("linear_units", 2048)
        num_blocks = decoder_conf.get("num_blocks", 6)
        dropout_rate = decoder_conf.get("dropout_rate", 0.1)
        activation_type = decoder_conf.get("activation_type", "relu")
        normalize_before = decoder_conf.get("normalize_before", True)
        use_output_layer = decoder_conf.get("use_output_layer", True)
        positional_dropout_rate = decoder_conf.get("positional_dropout_rate", 0.1)
        attention_heads = decoder_conf.get("attention_heads", 4)
        self_attention_dropout_rate = decoder_conf.get("self_attention_dropout_rate", 0.0)
        src_attention_dropout_rate = decoder_conf.get("src_attention_dropout_rate", 0.0)
        src_attention = decoder_conf.get("src_attention", True)
        query_bias = decoder_conf.get("query_bias", True)
        key_bias = decoder_conf.get("key_bias", True)
        value_bias = decoder_conf.get("value_bias", True)
        n_kv_head = decoder_conf.get("n_kv_head", None)
        head_dim = decoder_conf.get("head_dim", None)
        mlp_type = decoder_conf.get("mlp_type", "position_wise_feed_forward")
        mlp_bias = decoder_conf.get("mlp_bias", True)
        n_expert = decoder_conf.get("n_expert", 8)
        n_expert_activated = decoder_conf.get("n_expert_activated", 2)
        norm_eps: float = decoder_conf.get("norm_eps", 1e-5)
        input_layer = decoder_conf.get("input_layer", "embed")
        layer_norm_type = decoder_conf.get("layer_norm_type", "layer_norm")

        attention_dim = encoder_output_size
        activation = FQDD_ACTIVATIONS[activation_type]()

        self.embed = torch.nn.Sequential(
            torch.nn.Identity() if input_layer == "no_pos" else torch.nn.Embedding(vocab_size, attention_dim),
            FQDD_EMBEDDINGS[input_layer](attention_dim, positional_dropout_rate),
        )

        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.normalize_before = normalize_before
        self.after_norm = FQDD_NORMALIZES[layer_norm_type](attention_dim, eps=norm_eps)
        self.use_output_layer = use_output_layer
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = torch.nn.Identity()
        self.num_blocks = num_blocks

        mlp_class = FQDD_MLPS[mlp_type]
        self.decoders = torch.nn.ModuleList([
            DecoderLayer(
                attention_dim,
                FQDD_ATTENTIONS["selfattn"](
                    attention_heads, attention_dim,
                    self_attention_dropout_rate, query_bias, key_bias,
                    value_bias, n_kv_head, head_dim),
                FQDD_ATTENTIONS["crossattn"](
                    attention_heads, attention_dim, src_attention_dropout_rate,
                    query_bias, key_bias, value_bias, n_kv_head,
                    head_dim) if src_attention else None,
                mlp_class(attention_dim,
                          linear_units,
                          dropout_rate,
                          activation,
                          mlp_bias,
                          n_expert=n_expert,
                          n_expert_activated=n_expert_activated),
                dropout_rate,
                normalize_before,
                norm_eps,
            ) for _ in range(self.num_blocks)
        ])

    def forward(
            self,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            ys_in_pad: torch.Tensor,
            ys_in_lens: torch.Tensor,
            r_ys_in_pad: torch.Tensor = torch.empty(0),
            reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                torch.tensor(0.0), in order to unify api with bidirectional decoder
                olens: (batch, )
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        tgt = ys_in_pad
        maxlen = tgt.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1),
                            device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        x, _ = self.embed(tgt)

        x = self.forward_layers(x, tgt_mask, memory, memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)
        olens = tgt_mask.sum(1)
        return x, torch.tensor(0.0), olens

    def forward_layers(self, x: torch.Tensor, tgt_mask: torch.Tensor,
                       memory: torch.Tensor,
                       memory_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory,
                                                     memory_mask)
        return x
