"""Utility layers for models."""

import torch


@torch.compile
class ResidualLayer(torch.nn.Module):
    """
    Wrap a module with a residual connection.

    See: https://stackoverflow.com/a/78480544
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)
