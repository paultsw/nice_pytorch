"""
Utilities for loading, rescaling, image processing.
"""
import torch


def unflatten_images(input_batch, depth, height, width):
    """
    Take a batch of images and unflatten into a DxHxW grid.
    Nearly an inverse of `flatten_images`. (`flatten_images` assumes a list of tensors, not a tensor.)
    
    Args:
    * input_batch: a tensor of dtype=float and shape (bsz, d*h*w).
    * depth: int
    * height: int
    * width: int
    """
    return input_batch.view(input_batch.shape[0], depth, height, width)


def rescale(x, lo, hi):
    """Rescale a tensor to [lo,hi]."""
    assert(lo < hi), "[rescale] lo={0} must be smaller than hi={1}".format(lo,hi)
    old_width = torch.max(x)-torch.min(x)
    old_center = torch.min(x) + (old_width / 2.)
    new_width = float(hi-lo)
    new_center = lo + (new_width / 2.)
    # shift everything back to zero:
    x = x - old_center
    # rescale to correct width:
    x = x * (new_width / old_width)
    # shift everything to the new center:
    x = x + new_center
    # return:
    return x


def l1_norm(mdl, include_bias=True):
    """Compute L1 norm on all the weights of mdl."""
    if include_bias:
        _norm = torch.tensor(0.0)
        for w in mdl.parameters():
            _norm = _norm + w.norm(p=1)
        return _norm
    else:
        _norm = torch.tensor(0.0)
        for w in mdl.parameters():
            if len(w.shape) > 1:
                _norm = _norm + w.norm(p=1)
        return _norm
