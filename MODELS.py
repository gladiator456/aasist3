"""Compatibility wrapper for the canonical model implementation.

Use MODELDEFINITION.Model for training because it returns (features, logits).
This module keeps the older class name available for scripts that expect logits
only, without maintaining a second, incompatible architecture.
"""
import torch.nn as nn
from MODELDEFINITION import Model


_DEFAULT_ARGS = {
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 1.0, 1.0],
}


class AASIST3_Complete_Model(nn.Module):
    """Legacy API wrapper; forward returns logits of shape [batch, 2]."""
    def __init__(self, config=None):
        super().__init__()
        args = dict(_DEFAULT_ARGS)
        if config:
            args.update({key: value for key, value in config.items() if key in args})
        self.model = Model(args)

    def forward(self, x):
        _, logits = self.model(x)
        return logits
