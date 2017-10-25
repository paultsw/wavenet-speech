"""
layernorm.py: LayerNorm module implementation, adapted for convolutions.

Credit to @jekbradbury for the implementation below; see
  https://github.com/pytorch/pytorch/issues/1959
"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    `features` is the dimensionality of the convolutional sequence, expected to be
    BCS format (== (batch, nchannels, sequence_length)).
    
    Layer-norming is done across the dimension indicated by `dim`.
    """
    def __init__(self, features, dim=1, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features).unsqueeze(0).unsqueeze(2))
        self.beta = nn.Parameter(torch.zeros(features).unsqueeze(0).unsqueeze(2))
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        mean = x.mean(self.dim, keepdim=True).expand_as(x)
        std = x.std(self.dim, keepdim=True).expand_as(x)
        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps) + self.beta.expand_as(x)
