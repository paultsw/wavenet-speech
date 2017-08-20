"""
Convolutional modules.

TODO: consider importing this from Seq2Seq/modules/causal_conv1d.py
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class CausalConv1d(nn.Module):
    """
    Define a causal Conv1d as a special Conv1d that shifts the outputs
    by a certain amount.

    This module preserves temporal resolution (i.e. the number of timesteps
    in the input and output sequences is invariant).
    """
    def __init__(in_channels, out_channels, kernel_width, dilation):
        """Create underlying causal convolution."""
        # run parent initialization:
        super(CausalConv1d, self).__init__()
        
        # save arguments as attributes:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_width
        self.dilation = dilation

        # single conv1d as submodule:
        self.conv1d = nn.Conv1d(in_channels, out_channels, )

        # compute padding/shift:
        self.padding = None # [FIX]
        self.shift = None # [FIX]

    def forward(self, seq):
        """
        Perform a padding operation on both sides and shift after
        the conv1d.
        """
        pass
