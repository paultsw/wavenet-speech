"""
layernorm.py: LayerNorm module implementation.

Credit to @jekbradbury for the implementation below; see
  https://github.com/pytorch/pytorch/issues/1959
"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features)).unsqueeze(0).unsqueeze(1)
        self.beta = nn.Parameter(torch.zeros(features)).unsqueeze(0).unsqueeze(1)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1).expand_as(x)
        std = x.std(-1).expand_as(x)
        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps) + self.beta.expand_as(x)

    def cuda(self):
        self.gamma = self.gamma.cuda()
        self.beta = self.beta.cuda()
