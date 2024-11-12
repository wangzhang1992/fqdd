import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Union, Tuple, Dict
from fqdd.models.CTC import CTC
from fqdd.nnets.base_utils import ACTIVATION_CLASSES
from fqdd.utils.common import load_json_cmvn, pad_list

# sys.path.insert(0, "./")
# from src.model.base import ACTIVATION_CLASSES

T_CACHE = Tuple[torch.Tensor, torch.Tensor]
IGNORE_ID = -1


def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor,
                ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1),
                                pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator / denominator).detach()


def remove_duplicates_and_blank(hyp: List[int],
                                blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


class DecodeResult:

    def __init__(self,
                 tokens: List[int],
                 score: float = 0.0,
                 confidence: float = 0.0,
                 tokens_confidence: List[float] = None,
                 times: List[int] = None,
                 nbest: List[List[int]] = None,
                 nbest_scores: List[float] = None,
                 nbest_times: List[List[int]] = None):
        """
        Args:
            tokens: decode token list
            score: the total decode score of this result
            confidence: the total confidence of this result, it's in 0~1
            tokens_confidence: confidence of each token
            times: timestamp of each token, list of (start, end)
            nbest: nbest result
            nbest_scores: score of each nbest
            nbest_times:
        """
        self.tokens = tokens
        self.score = score
        self.confidence = confidence
        self.tokens_confidence = tokens_confidence
        self.times = times
        self.nbest = nbest
        self.nbest_scores = nbest_scores
        self.nbest_times = nbest_times


class GlobalCMVN(torch.nn.Module):

    def __init__(self,
                 mean: torch.Tensor,
                 istd: torch.Tensor,
                 norm_var: bool = True):
        """
        Args:
            mean (torch.Tensor): mean stats
            istd (torch.Tensor): inverse std, std which is 1.0 / std
        """
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        # The buffer can be accessed from this module using self.mean
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (batch, max_len, feat_dim)

        Returns:
            (torch.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x


class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
    """

    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = False):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            target (torch.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size

        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom


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


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
            self,
            idim: int,
            hidden_units: int,
            dropout_rate: float,
            activation: torch.nn.Module = torch.nn.ReLU(),
            bias: bool = True,
            *dummy_args,
            **dummy_kwargs,
    ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


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
            self.act = ACTIVATION_CLASSES[gate_activation]()

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


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def subsequent_mask(
        size: int,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    arange = torch.arange(size, device=device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask


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


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int, torch.tensor): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """

        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int,
                          apply_dropout: bool = True) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        # How to subscript a Union type:
        #   https://github.com/pytorch/pytorch/issues/69434
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:  # scalar
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        else:  # for batched streaming decoding on GPU
            assert torch.max(offset) + size <= self.max_len
            index = offset.unsqueeze(1) + \
                    torch.arange(0, size).to(offset.device)  # B X T
            flag = index > 0
            # remove negative offset
            index = index * flag
            pos_emb = F.embedding(index, self.pe[0])  # B X T X d_model

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


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


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    if n_kv_head != None and n_kv_head != n_head
    see: https://arxiv.org/pdf/1911.02150.pdf
         https://arxiv.org/pdf/2305.13245.pdf

    Example:
        case 1: n_kv_head == None, head_dim == None, MultiHead attention (MHSA)
        case 2: n_kv_head=1, n_head = 16, MultiQuery attention (MQA)
        case 3: nv_kv_head=2, n_head = 16, GroupedQuery attention (GQA)

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 query_bias: bool = True,
                 key_bias: bool = True,
                 value_bias: bool = True,
                 n_kv_head: Optional[int] = None,
                 head_dim: Optional[int] = None):
        """Construct an MultiHeadedAttention object."""
        super().__init__()

        self.inner_dim = n_feat if head_dim is None else head_dim * n_head
        if n_kv_head is not None:
            assert head_dim is not None
            self.inner_kv_dim = head_dim * n_kv_head
            n_kv_head = n_kv_head
        else:
            self.inner_kv_dim = self.inner_dim
            n_kv_head = n_head
        # We assume d_v always equals d_k
        self.d_k = self.inner_dim // n_head
        assert self.d_k == self.inner_kv_dim // n_kv_head
        self.h = n_head
        self.h_kv = n_kv_head

        self.linear_q = nn.Linear(n_feat, self.inner_dim, bias=query_bias)
        self.linear_k = nn.Linear(n_feat, self.inner_kv_dim, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, self.inner_kv_dim, bias=value_bias)
        self.linear_out = nn.Linear(self.inner_dim, n_feat, bias=query_bias)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.dropout_rate = dropout_rate

    def _forward_linearx(self, name: str, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 3
        if name == 'query':
            x = self.linear_q(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h, self.d_k])
        elif name == 'key':
            x = self.linear_k(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h_kv, self.d_k])
        else:
            assert name == 'value'
            x = self.linear_v(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h_kv, self.d_k])

        # split last dim
        x = x.view(x_shape)
        x = x.transpose(-3, -2)  # (batch, ...,  head or head_kv, time, d_k)
        return x

    def forward_qkv(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, ..., time1, size).
            key (torch.Tensor): Key tensor (#batch, ..., time2, size).
            value (torch.Tensor): Value tensor (#batch, ..., time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, ..., n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, ..., n_head_kv, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, ..., n_head_kv, time2, d_k).

        """
        q = self._forward_linearx('query', query)
        k = self._forward_linearx('key', key)
        v = self._forward_linearx('value', value)
        return q, k, v

    def forward_attention(
            self,
            value: torch.Tensor,
            scores: torch.Tensor,
            mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, ..., n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, ..., n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, ..., time1, time2), (0, ..., 0, 0) means fake mask.

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        # NOTE(xcsong): When will `if mask.size(2) > 0` be True?
        #   1. onnx(16/4) [WHY? Because we feed real cache & real mask for the
        #           1st chunk to ease the onnx export.]
        #   2. pytorch training
        if mask.size(-1) > 0:  # time2 > 0
            mask = mask.unsqueeze(-3).eq(0)  # (batch, .., 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[..., :scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores.float(),
                                 dim=-1).type_as(value).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be False?
        #   1. onnx(16/-1, -1/-1, 16/0)
        #   2. jit (16/-1, -1/-1, 16/0, 16/4)
        else:
            attn = torch.softmax(scores.float(), dim=-1).type_as(
                value)  # (batch, ..., head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, ...,  head, time1, d_k)
        x = x.transpose(-3, -2).contiguous()  # [batch, ..., time1, head, d_k]
        x_shape = x.size()[:-2] + torch.Size([self.h * self.d_k])
        x = x.view(x_shape)  # (batch, ..., time1, d_model)
        return self.linear_out(x)  # (batch, ...,  time1, d_model)

    def _update_kv_and_cache(
            self, k: torch.Tensor, v: torch.Tensor,
            cache: T_CACHE) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE]:
        new_cache = cache
        if not self.training:
            # NOTE(xcsong):
            #   when export onnx model, for 1st chunk, we feed
            #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
            #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
            #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
            #       and we will always do splitting and
            #       concatnation(this will simplify onnx export). Note that
            #       it's OK to concat & split zero-shaped tensors(see code below).
            #   when export jit  model, for 1st chunk, we always feed
            #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
            # >>> a = torch.ones((1, 2, 0, 4))
            # >>> b = torch.ones((1, 2, 3, 4))
            # >>> c = torch.cat((a, b), dim=2)
            # >>> torch.equal(b, c)        # True
            # >>> d = torch.split(a, 2, dim=-1)
            # >>> torch.equal(d[0], d[1])  # True
            key_cache, value_cache = cache
            if key_cache.size(0) > 0:
                k = torch.cat([key_cache, k], dim=2)
            if value_cache.size(0) > 0:
                v = torch.cat([value_cache, v], dim=2)
            # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
            #   non-trivial to calculate `next_cache_start` here.
            # new_cache = torch.cat((k, v), dim=-1) if not self.training else cache
            new_cache = (k, v)
        # for multi query or multi group attention
        if self.h_kv != self.h and self.h_kv != 1:
            # NOTE: onnxruntime issues:
            #     https://github.com/wenet-e2e/wenet/issues/2517
            # k = torch.repeat_interleave(
            #     k,
            #     self.h // self.h_kv,
            #     dim=-3,
            # )
            # v = torch.repeat_interleave(
            #     v,
            #     self.h // self.h_kv,
            #     dim=-3,
            # )
            n_repeat = self.h // self.h_kv
            k_shape = k.size()
            k = k.unsqueeze(-3).expand(
                k_shape[:-2] + torch.Size([n_repeat]) +
                k_shape[-2:]).reshape(k_shape[:-3] +
                                      torch.Size([self.h_kv * n_repeat]) +
                                      k_shape[-2:])
            v_shape = v.size()
            v = v.unsqueeze(-3).expand(
                v_shape[:-2] + torch.Size([n_repeat]) +
                v_shape[-2:]).reshape(v_shape[:-3] +
                                      torch.Size([self.h_kv * n_repeat]) +
                                      v_shape[-2:])
        return k, v, new_cache

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            pos_emb: torch.Tensor = torch.empty(0),
            cache: T_CACHE = (torch.zeros(0, 0, 0, 0), torch.zeros(0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, T_CACHE]:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        q, k, v = self.forward_qkv(query, key, value)
        k, v, new_cache = self._update_kv_and_cache(k, v, cache)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class MultiHeadedCrossAttention(MultiHeadedAttention):

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 query_bias: bool = True,
                 key_bias: bool = True,
                 value_bias: bool = True,
                 n_kv_head: Optional[int] = None,
                 head_dim: Optional[int] = None):
        super().__init__(n_head, n_feat, dropout_rate, query_bias, key_bias,
                         value_bias, n_kv_head, head_dim)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            pos_emb: torch.Tensor = torch.empty(0),
            cache: T_CACHE = (torch.zeros((0, 0, 0, 0)), torch.zeros((0, 0, 0, 0)))
    ) -> Tuple[torch.Tensor, T_CACHE]:
        del pos_emb
        key_cache, value_cache = cache
        assert key_cache.size(0) == value_cache.size(0)
        if key_cache.size(0) > 0:
            assert not self.training
            q = self._forward_linearx('query', query)
            k, v = key_cache, value_cache

        else:
            q, k, v = self.forward_qkv(query, key, value)
        new_cache = (k, v) if not self.training else cache
        # for multi query or multi groups attention
        if self.h_kv != self.h and self.h_kv != 1:
            k = torch.repeat_interleave(
                k,
                self.h // self.h_kv,
                dim=-3,
            )
            v = torch.repeat_interleave(
                v,
                self.h // self.h_kv,
                dim=-3,
            )
        B = query.size(0)
        Beams = 1
        if B != k.size(0):
            assert not self.training
            Beams = B // k.size(0)
            B = k.size(0)
            q = q.view(B, Beams, q.size(-3), q.size(-2), q.size(-1))
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            mask = mask.unsqueeze(1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        output = self.forward_attention(v, scores, mask)

        if query.size(0) != B:
            assert not self.training
            output_shape = torch.Size([B * Beams]) + output.size()[2:]
            output = output.view(output_shape)
        return output, new_cache


class BaseSubsampling(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class Conv2dSubsampling2(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling4 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, odim, 3, 2),
                                        torch.nn.ReLU())
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((idim - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 2
        # 2 = (3 - 1) * 1
        self.right_context = 2

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
            offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2]


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

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size, eps=norm_eps)
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
        mlp_class = PositionwiseFeedForward
        pos_emb_class = PositionalEncoding
        activation = ACTIVATION_CLASSES[activation_type]()
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
            mlp_bias,
            n_expert,
            n_expert_activated,
        )

        self.embed = Conv2dSubsampling2(
            input_size, output_size, dropout_rate,
            pos_emb_class(output_size, positional_dropout_rate),
            # pos_emb_class(output_size, output_size // attention_heads, positional_dropout_rate
        )

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
        mlp_bias = decoder_conf.get("mlp_bias", True, )
        n_expert = decoder_conf.get("n_expert", 8, )
        n_expert_activated = decoder_conf.get("n_expert_activated", 2)
        norm_eps: float = decoder_conf.get("norm_eps", 1e-5)

        attention_dim = encoder_output_size
        activation = ACTIVATION_CLASSES[activation_type]()

        self.embed = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, attention_dim),
            PositionalEncoding(attention_dim, positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(attention_dim, eps=norm_eps)
        self.use_output_layer = use_output_layer
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = torch.nn.Identity()
        self.num_blocks = num_blocks

        mlp_class = PositionwiseFeedForward
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

        ys_in_pad, ys_out_pad = self.add_sos_eos(padding_ys, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_lens + 1

        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask, ys_in_pad, ys_in_lens)
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

    def add_sos_eos(self, ys_pad: torch.Tensor, sos: int, eos: int,
                    ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add <sos> and <eos> labels.

        Args:
            self (torch.Tensor): batch of padded target sequences (B, Lmax)
            sos (int): index of <sos>
            eos (int): index of <eeos>
            ignore_id (int): index of padding

        Returns:
            ys_in (torch.Tensor) : (B, Lmax + 1)
            ys_out (torch.Tensor) : (B, Lmax + 1)

        Examples:
            >>> sos_id = 10
            >>> eos_id = 11
            >>> ignore_id = -1
            >>> self
            tensor([[ 1,  2,  3,  4,  5],
                    [ 4,  5,  6, -1, -1],
                    [ 7,  8,  9, -1, -1]], dtype=torch.int32)
            >>> ys_in,ys_out=add_sos_eos(self, sos_id , eos_id, ignore_id)
            >>> ys_in
            tensor([[10,  1,  2,  3,  4,  5],
                    [10,  4,  5,  6, -1, -1],
                    [10,  7,  8,  9, -1, -1]])
            >>> ys_out
            tensor([[ 1,  2,  3,  4,  5, 11],
                    [ 4,  5,  6, 11, -1, -1],
                    [ 7,  8,  9, 11, -1, -1]])
        """
        _sos = torch.tensor([sos],
                            dtype=torch.long,
                            requires_grad=False,
                            device=ys_pad.device)
        _eos = torch.tensor([eos],
                            dtype=torch.long,
                            requires_grad=False,
                            device=ys_pad.device)
        ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
        ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, _eos], dim=0) for y in ys]

        return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)

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
