import torch
import torch.nn as nn

from typing import List, Optional, Union, Tuple, Dict
from fqdd.modules.CTC import CTC
from fqdd.module_nnets.attentions import MultiHeadedAttention, MultiHeadedCrossAttention
from fqdd.decoders.search import DecodeResult
from fqdd.module_nnets.model_utils import FQDD_EMBEDDINGS, FQDD_MLPS, FQDD_SUBSAMPLES
from fqdd.element_nnets.base_utils import FQDD_ACTIVATIONS, FQDD_NORMALIZES
from fqdd.modules.losses import LabelSmoothingLoss
from fqdd.text.tokenize_utils import remove_duplicates_and_blank, add_sos_eos, reverse_pad_list
from fqdd.utils.common import load_json_cmvn, GlobalCMVN, th_accuracy
from fqdd.utils.mask import make_pad_mask, subsequent_mask

# sys.path.insert(0, "./")
# from src.model.base import ACTIVATION_CLASSES

T_CACHE = Tuple[torch.Tensor, torch.Tensor]
IGNORE_ID = -1


class EBranchformerEncoderLayer(torch.nn.Module):
    """E-Branchformer encoder layer module.

    Args:
        size (int): model dimension
        attn: standard self-attention or efficient attention
        cgmlp: ConvolutionalGatingMLP
        feed_forward: feed-forward module, optional
        feed_forward: macaron-style feed-forward module, optional
        dropout_rate (float): dropout probability
        merge_conv_kernel (int): kernel size of the depth-wise conv in merge module
    """

    def __init__(
            self,
            size: int,
            attn: torch.nn.Module,
            cgmlp: torch.nn.Module,
            feed_forward: Optional[torch.nn.Module],
            feed_forward_macaron: Optional[torch.nn.Module],
            dropout_rate: float,
            merge_conv_kernel: int = 3,
            causal: bool = True,
            stochastic_depth_rate=0.0,
    ):
        super().__init__()

        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp

        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.ff_scale = 1.0
        if self.feed_forward is not None:
            self.norm_ff = nn.LayerNorm(size)
        if self.feed_forward_macaron is not None:
            self.ff_scale = 0.5
            self.norm_ff_macaron = nn.LayerNorm(size)

        self.norm_mha = nn.LayerNorm(size)  # for the MHA module
        self.norm_mlp = nn.LayerNorm(size)  # for the MLP module
        # for the final output of the block
        self.norm_final = nn.LayerNorm(size)

        self.dropout = torch.nn.Dropout(dropout_rate)

        if causal:
            padding = 0
            self.lorder = merge_conv_kernel - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (merge_conv_kernel - 1) % 2 == 0
            padding = (merge_conv_kernel - 1) // 2
            self.lorder = 0
        self.depthwise_conv_fusion = torch.nn.Conv1d(
            size + size,
            size + size,
            kernel_size=merge_conv_kernel,
            stride=1,
            padding=padding,
            groups=size + size,
            bias=True,
        )
        self.merge_proj = torch.nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def _forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            pos_emb: torch.Tensor,
            mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            att_cache: T_CACHE = (torch.zeros(
                (0, 0, 0, 0)), torch.zeros(0, 0, 0, 0)),
            cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
            stoch_layer_coeff: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:

        if self.feed_forward_macaron is not None:
            residual = x
            x = self.norm_ff_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))

        # Two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        x1 = self.norm_mha(x1)
        x_att, new_att_cache = self.attn(x1, x1, x1, mask, pos_emb, att_cache)
        x1 = self.dropout(x_att)

        # Branch 2: convolutional gating mlp
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        x2 = self.norm_mlp(x2)
        x2, new_cnn_cache = self.cgmlp(x2, mask_pad, cnn_cache)
        x2 = self.dropout(x2)

        # Merge two branches
        x_concat = torch.cat([x1, x2], dim=-1)
        x_tmp = x_concat.transpose(1, 2)
        if self.lorder > 0:
            x_tmp = nn.functional.pad(x_tmp, (self.lorder, 0), "constant", 0.0)
            assert x_tmp.size(2) > self.lorder
        x_tmp = self.depthwise_conv_fusion(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)
        x = x + stoch_layer_coeff * self.dropout(
            self.merge_proj(x_concat + x_tmp))

        if self.feed_forward is not None:
            # feed forward module
            residual = x
            x = self.norm_ff(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward(x))

        x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            pos_emb: torch.Tensor,
            mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            att_cache: T_CACHE = (torch.zeros(
                (0, 0, 0, 0)), torch.zeros(0, 0, 0, 0)),
            cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (Union[Tuple, torch.Tensor]): Input tensor  (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time, time).
            pos_emb (torch.Tensor): positional encoding, must not be None
                for BranchformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in cgmlp layer
                (#batch=1, size, cache_t2)

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time.
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """

        stoch_layer_coeff = 1.0
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        if self.training:
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)
        return self._forward(x, mask, pos_emb, mask_pad, att_cache, cnn_cache,
                             stoch_layer_coeff)


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Convolutional Spatial Gating Unit (CSGU)."""

    def __init__(
            self,
            size: int,
            kernel_size: int,
            dropout_rate: float,
            use_linear_after_conv: bool,
            gate_activation: str,
            causal: bool = True,
    ):
        super().__init__()

        # split input channels
        n_channels = size // 2
        self.norm = nn.LayerNorm(n_channels)
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.conv = torch.nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size,
            1,
            padding,
            groups=n_channels,
        )
        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        if gate_activation == "identity":
            self.act = torch.nn.Identity()
        else:
            self.act = FQDD_ACTIVATIONS[gate_activation]()

        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        torch.nn.init.normal_(self.conv.weight, std=1e-6)
        torch.nn.init.ones_(self.conv.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)

    def forward(
            self, x: torch.Tensor, cache: torch.Tensor = torch.zeros((0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward method

        Args:
            x (torch.Tensor): (batch, time, channels)
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.

        Returns:
            out (torch.Tensor): (batch, time, channels/2)
        """

        x_r, x_g = x.chunk(2, dim=-1)
        # exchange the temporal dimension and the feature dimension
        x_g = x_g.transpose(1, 2)  # (#batch, channels, time)

        if self.lorder > 0:
            if cache.size(2) == 0:  # cache_t == 0
                x_g = nn.functional.pad(x_g, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x_g.size(0)  # equal batch
                assert cache.size(1) == x_g.size(1)  # equal channel
                x_g = torch.cat((cache, x_g), dim=2)
            assert (x_g.size(2) > self.lorder)
            new_cache = x_g[:, :, -self.lorder:]
        else:
            # It's better we just return None if no cache is required,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = torch.zeros((0, 0, 0),
                                    dtype=x_g.dtype,
                                    device=x_g.device)

        x_g = x_g.transpose(1, 2)
        x_g = self.norm(x_g)  # (N, T, D/2)
        x_g = self.conv(x_g.transpose(1, 2)).transpose(1, 2)  # (N, T, D/2)
        if self.linear is not None:
            x_g = self.linear(x_g)

        x_g = self.act(x_g)
        out = x_r * x_g  # (N, T, D/2)
        out = self.dropout(out)
        return out, new_cache


class ConvolutionalGatingMLP(torch.nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(
            self,
            size: int,
            linear_units: int,
            kernel_size: int,
            dropout_rate: float,
            use_linear_after_conv: bool,
            gate_activation: str,
            causal: bool = True,
    ):
        super().__init__()

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, linear_units), torch.nn.GELU())
        self.csgu = ConvolutionalSpatialGatingUnit(
            size=linear_units,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
            gate_activation=gate_activation,
            causal=causal,
        )
        self.channel_proj2 = torch.nn.Linear(linear_units // 2, size)

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            cache: torch.Tensor = torch.zeros((0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward method

        Args:
            x (torch.Tensor): (batch, time, channels)
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask. Not used yet
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.

        Returns:
            out (torch.Tensor): (batch, time, channels/2)
        """

        xs_pad = x

        # size -> linear_units
        xs_pad = self.channel_proj1(xs_pad)

        # linear_units -> linear_units/2
        xs_pad, new_cnn_cache = self.csgu(xs_pad, cache)

        # linear_units/2 -> size
        xs_pad = self.channel_proj2(xs_pad)

        out = xs_pad

        return out, new_cnn_cache


def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret


# modify from : https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/layer_drop.py # noqa
class LayerDropModuleList(torch.nn.ModuleList):
    """
    A LayerDrop implementation based on :class:`torch.nn.ModuleList`.

    We refresh the choice of which layers to drop every time we iterate
    over the LayerDropModuleList instance. During evaluation we always
    iterate over all layers.

    Usage::

        layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
        for layer in layers:  # this might iterate over layers 1 and 3
            x = layer(x)
        for layer in layers:  # this might iterate over all layers
            x = layer(x)
        for layer in layers:  # this might not iterate over any layers
            x = layer(x)

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add

    Limitations:
        1 can work with ddp when layer's gradient checkpoint disabled
        2 can't work with ddp when layer's gradient checkpoint enables
        3 can work with fsdp
        4 can work with deepspeed
    """

    def __init__(self, p: List[float], modules=None):
        super().__init__(modules)
        assert len(p) == len(self)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p[i]):
                yield m


class EBranchformerEncoder(nn.Module):
    """E-Branchformer encoder module."""

    def __init__(
            self,
            encoder_conf,
            use_cmvn: bool = True,
            cmvn_file: str = None
    ):
        super(EBranchformerEncoder, self).__init__()
        input_size = encoder_conf.get("input_size")
        output_size = encoder_conf.get("output_size", 256)
        attention_heads = encoder_conf.get("attention_heads", 4)
        linear_units = encoder_conf.get("linear_units", 2048)
        cgmlp_linear_units = encoder_conf.get("cgmlp_linear_units", 2048)
        cgmlp_conv_kernel = encoder_conf.get("cgmlp_conv_kernel", 31)
        use_linear_after_conv = encoder_conf.get("use_linear_after_conv", False)
        gate_activation = encoder_conf.get("gate_activation", "identity")
        num_blocks = encoder_conf.get("num_blocks", 12)
        dropout_rate = encoder_conf.get("dropout_rate", 0.1)
        attention_dropout_rate = encoder_conf.get("attention_dropout_rate", 0.0)
        normalize_before = encoder_conf.get("normalize_before", True)
        layer_norm_type = encoder_conf.get("layer_norm_type", "layer_norm")
        stochastic_depth_rate = encoder_conf.get("stochastic_depth_rate", 0.0)
        causal = encoder_conf.get("causal", False, )
        activation_type = encoder_conf.get("activation_type", "swish")
        positional_dropout_rate = encoder_conf.get("positional_dropout_rate", 0.1)
        merge_conv_kernel = encoder_conf.get("merge_conv_kernel", 3)
        use_ffn = encoder_conf.get("use_ffn", True)
        macaron_style = encoder_conf.get("macaron_style", True)
        query_bias = encoder_conf.get("query_bias", True)
        key_bias = encoder_conf.get("key_bias", True)
        value_bias = encoder_conf.get("value_bias", True)
        n_kv_head = encoder_conf.get("n_kv_head", None)
        head_dim = encoder_conf.get("head_dim", None)
        mlp_type = encoder_conf.get("mlp_type", "position_wise_feed_forward")
        mlp_bias = encoder_conf.get("mlp_bias", True)
        n_expert = encoder_conf.get("n_expert", 8)
        input_layer = encoder_conf.get("input_layer", "conv2d")
        n_expert_activated = encoder_conf.get("n_expert_activated", 2)
        pos_enc_layer_type = encoder_conf.get("pos_enc_layer_type", "abs_pos")
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

        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.normalize_before = normalize_before
        self.after_norm = FQDD_NORMALIZES[layer_norm_type](output_size, eps=norm_eps)
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

        self.embed = FQDD_SUBSAMPLES[input_layer](
            input_size, output_size, dropout_rate,
            pos_emb_class(output_size, positional_dropout_rate)
            if pos_enc_layer_type != 'rope_pos' else pos_emb_class(
                output_size, output_size //
                             attention_heads, positional_dropout_rate))

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks
        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})")

        self.encoders = LayerDropModuleList(
            p=stochastic_depth_rate,
            modules=[
                EBranchformerEncoderLayer(
                    output_size,
                    MultiHeadedAttention(
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


class DecoderLayer(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Inter-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
            If `None` is passed, Inter-attention is not used, such as
            CIF, GPT, and other decoder only model.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
    """

    def __init__(
            self,
            size: int,
            self_attn: nn.Module,
            src_attn: Optional[nn.Module],
            feed_forward: nn.Module,
            dropout_rate: float,
            normalize_before: bool = True,
            norm_eps: float = 1e-5,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        self.norm1 = nn.LayerNorm(size, eps=norm_eps)
        self.norm2 = nn.LayerNorm(size, eps=norm_eps)
        self.norm3 = nn.LayerNorm(size, eps=norm_eps)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before

    def forward(
            self,
            tgt: torch.Tensor,
            tgt_mask: torch.Tensor,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            cache: Optional[Dict[str, Optional[T_CACHE]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory
                (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            cache (torch.Tensor): cached tensors.
                (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        if cache is not None:
            att_cache = cache['self_att_cache']
            cross_att_cache = cache['cross_att_cache']
        else:
            att_cache, cross_att_cache = None, None

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if att_cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
            att_cache = (torch.empty(0, 0, 0, 0), torch.empty(0, 0, 0, 0))
        else:
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]

        x, new_att_cache = self.self_attn(
            tgt_q,
            tgt_q,
            tgt_q,
            tgt_q_mask,
            cache=att_cache,
        )
        if cache is not None:
            cache['self_att_cache'] = new_att_cache
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.norm1(x)

        if self.src_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm2(x)
            if cross_att_cache is None:
                cross_att_cache = (torch.empty(0, 0, 0,
                                               0), torch.empty(0, 0, 0, 0))
            x, new_cross_cache = self.src_attn(x,
                                               memory,
                                               memory,
                                               memory_mask,
                                               cache=cross_att_cache)
            if cache is not None:
                cache['cross_att_cache'] = new_cross_cache
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        return x, tgt_mask, memory, memory_mask


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
        attention_heads = decoder_conf.get("attention_heads", 4, )
        linear_units = decoder_conf.get("linear_units", 2048, )
        num_blocks = decoder_conf.get("num_blocks", 6, )
        dropout_rate = decoder_conf.get("dropout_rate", 0.1, )
        activation_type = decoder_conf.get("activation_type", "relu")
        self_attention_dropout_rate = decoder_conf.get("self_attention_dropout_rate", 0.0, )
        src_attention_dropout_rate = decoder_conf.get("src_attention_dropout_rate", 0.0, )
        normalize_before = decoder_conf.get("normalize_before", True, )
        use_output_layer = decoder_conf.get("use_output_layer", True, )
        positional_dropout_rate = decoder_conf.get("positional_dropout_rate", 0.1, )
        query_bias = decoder_conf.get("query_bias", True, )
        key_bias = decoder_conf.get("key_bias", True, )
        value_bias = decoder_conf.get("value_bias", True, )
        n_kv_head = decoder_conf.get("n_kv_head", None, )
        head_dim = decoder_conf.get("head_dim", None, )
        mlp_type = decoder_conf.get("mlp_type", "position_wise_feed_forward")
        mlp_bias = decoder_conf.get("mlp_bias", True, )
        n_expert = decoder_conf.get("n_expert", 8, )
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
                MultiHeadedAttention(
                    attention_heads, attention_dim,
                    self_attention_dropout_rate, query_bias, key_bias,
                    value_bias, n_kv_head, head_dim),
                MultiHeadedCrossAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate,
                    query_bias, key_bias, value_bias, n_kv_head,
                    head_dim),
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


class EBranchformer(nn.Module):
    def __init__(
            self,
            model_conf,
    ):
        super(EBranchformer, self).__init__()

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

        self.encoder = EBranchformerEncoder(encoder_conf, use_cmvn, cmvn_file)
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
