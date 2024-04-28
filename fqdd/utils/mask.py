import torch 
import numpy as np


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    '''
    Args:
        lengths (torch.Tensor): Batch of length (B,)
    return:
        torch.Tensor: mask tensor containing indices of padded part.
    Examples:
        >>> lengths = [5, 3, 2]
        >>> mask_pad_mask(lengths)
        masks = [[0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    '''
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
