"""Compatibility wrapper for the canonical model implementation.

Use MODELDEFINITION.Model for training because it returns (features, logits).
This module keeps the older class name available for scripts that expect logits
only, without maintaining a second, incompatible architecture.
"""
import torch.nn as nn
from MODELDEFINITION import Model, DEFAULT_D_ARGS


_DEFAULT_ARGS = dict(DEFAULT_D_ARGS)


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
