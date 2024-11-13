from typing import Tuple

import torch.nn

import torch

from fqdd.models.conformer.encoder_layer import ConformerEncoderLayer
from fqdd.module_nnets.model_utils import FQDD_MLPS, FQDD_SUBSAMPLES, FQDD_EMBEDDINGS, FQDD_ATTENTIONS
from fqdd.element_nnets.CNN import ConvolutionModule
from fqdd.element_nnets.base_utils import FQDD_ACTIVATIONS, FQDD_NORMALIZES
from fqdd.utils.common import load_json_cmvn, GlobalCMVN
from fqdd.utils.mask import make_pad_mask


class ConformerEncoder(torch.nn.Module):
    """Conformer encoder module."""

    def __init__(
            self,
            encoder_conf,
            use_cmvn: bool = False,
            cmvn_file: str = None
    ):
        """Construct ConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
            key_bias: whether use bias in attention.linear_k, False for whisper models.
        """

        super(ConformerEncoder, self).__init__()

        input_size = encoder_conf.get("input_size")
        output_size = encoder_conf.get("output_size", 256)
        attention_heads = encoder_conf.get("attention_heads", 4)
        linear_units = encoder_conf.get("linear_units", 2048)
        num_blocks = encoder_conf.get("num_blocks", 12)
        dropout_rate = encoder_conf.get("dropout_rate", 0.1)
        attention_dropout_rate = encoder_conf.get("attention_dropout_rate", 0.0)
        positional_dropout_rate = encoder_conf.get("positional_dropout_rate", 0.1)
        input_layer = encoder_conf.get("input_layer", "conv2d")
        pos_enc_layer_type = encoder_conf.get("pos_enc_layer_type", "rel_pos")
        normalize_before = encoder_conf.get("normalize_before", True)
        positionwise_conv_kernel_size = encoder_conf.get("positionwise_conv_kernel_size", 1)
        macaron_style = encoder_conf.get("macaron_style", True)
        selfattention_layer_type = encoder_conf.get("selfattention_layer_type", "rel_selfattn")
        activation_type = encoder_conf.get("activation_type", "swish")
        use_cnn_module = encoder_conf.get("use_cnn_module")
        cnn_module_kernel = encoder_conf.get("cnn_module_kernel")
        causal = encoder_conf.get("causal", False)
        cnn_module_norm = encoder_conf.get("cnn_module_norm", "batch_norm")
        query_bias = encoder_conf.get("query_bias", True)
        key_bias = encoder_conf.get("key_bias", True)
        value_bias = encoder_conf.get("value_bias", True)
        conv_bias = encoder_conf.get("conv_bias", True)
        layer_norm_type = encoder_conf.get("layer_norm_type", "layer_norm")
        norm_eps: float = encoder_conf.get("norm_eps", 1e-5)
        n_kv_head = encoder_conf.get("n_kv_head", None)
        head_dim = encoder_conf.get("head_dim", None)
        mlp_type = encoder_conf.get("mlp_type", "position_wise_feed_forward")
        mlp_bias = encoder_conf.get("mlp_bias", True)
        n_expert = encoder_conf.get("n_expert", 8)
        n_expert_activated = encoder_conf.get("n_expert_activated", 2)
        use_cmvn = use_cmvn
        cmvn_file = cmvn_file

        self._output_size = output_size

        self.global_cmvn = None
        if use_cmvn:
            mean, std = load_json_cmvn(cmvn_file)
            mean = torch.from_numpy(mean).float()
            std = torch.from_numpy(std).float()
            self.global_cmvn = GlobalCMVN(mean, std)

        self.normalize_before = normalize_before
        self.after_norm = FQDD_NORMALIZES[layer_norm_type](output_size, eps=norm_eps)

        activation = FQDD_ACTIVATIONS[activation_type]()
        pos_emb_class = FQDD_EMBEDDINGS[pos_enc_layer_type]
        self.embed = FQDD_SUBSAMPLES[input_layer](
            input_size, output_size, dropout_rate,
            pos_emb_class(output_size, positional_dropout_rate)
            if pos_enc_layer_type != 'rope_pos' else pos_emb_class(
                output_size, output_size //
                             attention_heads, positional_dropout_rate))
        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            query_bias,
            key_bias,
            value_bias,
            n_kv_head,
            head_dim
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
            mlp_bias,
            n_expert,
            n_expert_activated,
        )
        # convolution module definition
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal, conv_bias)

        mlp_class = FQDD_MLPS[mlp_type]
        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                FQDD_ATTENTIONS[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                mlp_class(*positionwise_layer_args),
                mlp_class(*positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                layer_norm_type=layer_norm_type,
                norm_eps=norm_eps,
            ) for _ in range(num_blocks)
        ])

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs: torch.Tensor,
            xs_lens: torch.Tensor,
            decoding_chunk_size: int = 0,
            num_decoding_left_chunks: int = -1,
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
        chunk_masks = masks

        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                       pos_emb: torch.Tensor,
                       mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs
