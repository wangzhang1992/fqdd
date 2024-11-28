from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn

from fqdd.modules.attentions import T_CACHE


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
            cgmlp: torch.nn.Module,
            dropout_rate: float,
            merge_conv_kernel: int = 3,
            causal: bool = True,
            normalize_before: bool = True,
            norm_eps: float = 1e-5,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.cgmlp = cgmlp
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
            cache: Optional[Dict[str, Optional[T_CACHE]]] = None,
            cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
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

        Parameters
        ----------
        cnn_cache

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

        tgt_q1 = tgt_q
        tgt_q2 = tgt_q

        tgt_q1, new_att_cache = self.self_attn(
            tgt_q1,
            tgt_q1,
            tgt_q1,
            tgt_q_mask,
            cache=att_cache,
        )
        if cache is not None:
            cache['self_att_cache'] = new_att_cache

        tgt_q2 = self.norm_mlp(tgt_q2)
        tgt_q2, new_cnn_cache = self.cgmlp(tgt_q2, tgt_q_mask, cnn_cache)
        tgt_q2 = self.dropout(tgt_q2)
        x_concat = torch.cat([tgt_q1, tgt_q2], dim=-1)
        x_tmp = x_concat.transpose(1, 2)
        if self.lorder > 0:
            x_tmp = nn.functional.pad(x_tmp, (self.lorder, 0), "constant", 0.0)
            assert x_tmp.size(2) > self.lorder
        x_tmp = self.depthwise_conv_fusion(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)

        x = residual + self.dropout(self.merge_proj(x_concat + x_tmp))

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
