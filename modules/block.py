"""
Definition of a skip-connection residual block, the base component of
the WaveNet architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.conv_modules import CausalConv1d
from collections import OrderedDict

### ===== ===== ===== ===== Skip-Connection residual block.
class ResidualBlock(nn.Module):
    """
    TBD: explanation
    """
    def __init__(self, *args, **kwargs):
        """Residual Block constructor"""
        ### parent constructor:
        super(ResidualBlock, self).__init__()

        ### attributes:
        # (tbd)

        ### submodules:
        self.dilated_causal_conv1d = CausalConv1d(None) # FIX ARGS
        self.gated_activation = GatedActivationUnit()
        self.conv1x1 = nn.Conv1d(kernel_size=1) # FIX ARGS
        self.residual_proj = nn.Linear(in_channels + out_channels, out_channels)
        

    def forward(self, seq):
        """
        Apply a residual block.
        
        Args:
        * seq
        
        Returns: (conv1x1_out, block_out) where:
        * conv1x1_out: the output after the conv1x1 layer.
        * block_out: residual block final output, after stacking `conv1x1_out`
          with `seq` and applying `residual_proj()` to each timestep.
        """
        pass


### ===== ===== ===== ===== Gated Activation Unit.
class GatedActivationUnit(nn.Module):
    """
    Stateless module implementing a gated activation of two
    sequences of same temporal dimension.

    Returns Tanh(x) .* Sigmoid(y)
    """
    def forward(self, x, y):
        return torch.mul(F.tanh(x_seq), F.sigmoid(y))

    def __repr__(self):
        return self.__class__.__name__ + ' ()'
