from typing import Tuple

import torch
import torch.nn as nn

from fqdd.nnets.base_utils import FQDD_NORMALIZES, FQDD_ACTIVATIONS
from fqdd.models.ebranchformer.encoder_layer import ConvolutionalGatingMLP, EBranchformerEncoderLayer
from fqdd.modules.model_utils import FQDD_MLPS, FQDD_EMBEDDINGS, FQDD_SUBSAMPLES, LayerDropModuleList, FQDD_ATTENTIONS
from fqdd.utils.common import load_json_cmvn, GlobalCMVN
from fqdd.utils.mask import make_pad_mask


class EBranchformerEncoder(nn.Module):
    """E-Branchformer encoder module."""

    def __init__(
            self,
            encoder_conf,
            use_cmvn: bool = False,
            cmvn_file: str = None
    ):
        super(EBranchformerEncoder, self).__init__()
        input_size = encoder_conf.get("input_size")
        output_size = encoder_conf.get("output_size", 256)
        attention_heads = encoder_conf.get("attention_heads", 4)
        linear_units = encoder_conf.get("linear_units", 2048)
        selfattention_layer_type = encoder_conf.get("selfattention_layer_type", "rel_selfattn")
        pos_enc_layer_type = encoder_conf.get("pos_enc_layer_type", "rel_pos")
        activation_type = encoder_conf.get("activation_type", "swish")
        cgmlp_linear_units = encoder_conf.get("cgmlp_linear_units", 2048)
        cgmlp_conv_kernel = encoder_conf.get("cgmlp_conv_kernel", 31)
        use_linear_after_conv = encoder_conf.get("use_linear_after_conv", False)
        gate_activation = encoder_conf.get("gate_activation", "identity")
        num_blocks = encoder_conf.get("num_blocks", 12)
        dropout_rate = encoder_conf.get("dropout_rate", 0.1)
        positional_dropout_rate = encoder_conf.get("positional_dropout_rate", 0.1)
        attention_dropout_rate = encoder_conf.get("attention_dropout_rate", 0.0)
        input_layer = encoder_conf.get("input_layer", "conv2d")
        stochastic_depth_rate = encoder_conf.get("stochastic_depth_rate", 0.0)
        causal = encoder_conf.get("causal", False)
        merge_conv_kernel = encoder_conf.get("merge_conv_kernel", 3)
        normalize_before = encoder_conf.get("normalize_before", True)
        use_ffn = encoder_conf.get("use_ffn", True)
        macaron_style = encoder_conf.get("macaron_style", True)
        query_bias = encoder_conf.get("query_bias", True)
        key_bias = encoder_conf.get("key_bias", True)
        value_bias = encoder_conf.get("value_bias", True)
        layer_norm_type = encoder_conf.get("layer_norm_type", "layer_norm")
        n_kv_head = encoder_conf.get("n_kv_head", None)
        head_dim = encoder_conf.get("head_dim", None)
        mlp_type = encoder_conf.get("mlp_type", "position_wise_feed_forward")
        mlp_bias = encoder_conf.get("mlp_bias", True)
        n_expert = encoder_conf.get("n_expert", 8)
        n_expert_activated = encoder_conf.get("n_expert_activated", 2)
        norm_eps: float = encoder_conf.get("norm_eps", 1e-5)
        use_cmvn = use_cmvn
        cmvn_file = cmvn_file

        self._output_size = output_size

        self.global_cmvn = None
        if use_cmvn:
            mean, std = load_json_cmvn(cmvn_file)
            mean = torch.from_numpy(mean).float()
            std = torch.from_numpy(std).float()
            self.global_cmvn = GlobalCMVN(mean, std)

        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            query_bias,
            key_bias,
            value_bias,
            n_kv_head,
            head_dim,
        )

        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (output_size, cgmlp_linear_units, cgmlp_conv_kernel,
                            dropout_rate, use_linear_after_conv,
                            gate_activation, causal)

        # feed-forward module definition
        mlp_class = FQDD_MLPS[mlp_type]
        pos_emb_class = FQDD_EMBEDDINGS[pos_enc_layer_type]
        activation = FQDD_ACTIVATIONS[activation_type]()
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
            mlp_bias,
            n_expert,
            n_expert_activated,
        )

        self.normalize_before = normalize_before
        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks
        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})")

        self.embed = FQDD_SUBSAMPLES[input_layer](
            input_size, output_size, dropout_rate,
            pos_emb_class(output_size, positional_dropout_rate)
            if pos_enc_layer_type != 'rope_pos' else pos_emb_class(
                output_size, output_size //
                             attention_heads, positional_dropout_rate))

        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.after_norm = FQDD_NORMALIZES[layer_norm_type](output_size, eps=norm_eps)
        self.encoders = LayerDropModuleList(
            p=stochastic_depth_rate,
            modules=[
                EBranchformerEncoderLayer(
                    output_size,
                    FQDD_ATTENTIONS[selfattention_layer_type](
                        *encoder_selfattn_layer_args),
                    cgmlp_layer(*cgmlp_layer_args),
                    mlp_class(*positionwise_layer_args) if use_ffn else None,
                    mlp_class(*positionwise_layer_args)
                    if use_ffn and macaron_style else None,
                    dropout_rate,
                    merge_conv_kernel=merge_conv_kernel,
                    causal=causal,
                    stochastic_depth_rate=stochastic_depth_rate[lnum],
                ) for lnum in range(num_blocks)
            ])
        # print(self.encoders)

    def forward_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                       pos_emb: torch.Tensor,
                       mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    def forward(
            self,
            xs: torch.Tensor,
            xs_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = mask_pad

        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)

        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks
