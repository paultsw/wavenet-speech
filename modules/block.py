"""
Definition of a skip-connection residual block, the base component of
the WaveNet architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.conv_ops import CausalConv1d, NonCausalConv1d, reshape_in, reshape_out
from collections import OrderedDict

### ===== ===== ===== ===== Skip-Connection residual block.
class ResidualBlock(nn.Module):
    """
    A residual block consisting of causal 1D convolutions, an internal nonlinearity given by
    the gated activation unit, and a conv1x1.

    Returns additive residual output and a skip connection.
    """
    def __init__(self, in_channels, out_channels, kernel_width, dilation,
                 causal=True, conditioning=None):
        """
        Residual Block constructor.
        
        The 'causal' flag determines whether the internal convolutions are causal or standard.
        Will (eventually) allow for optional conditioning on a global input.
        """
        ### parent constructor:
        super(ResidualBlock, self).__init__()

        ### attributes:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_width
        self.dilation = dilation
        self.causal = causal
        self.conditioning = not (conditioning == None)

        ### submodules:
        if causal:
            self.conv_tanh = CausalConv1d(in_channels, out_channels, kernel_width, dilation=dilation)
            self.conv_sigmoid = CausalConv1d(in_channels, out_channels, kernel_width, dilation=dilation)
        else:
            self.conv_tanh = NonCausalConv1d(in_channels, out_channels, kernel_width, dilation=dilation)
            self.conv_sigmoid = NonCausalConv1d(in_channels, out_channels, kernel_width, dilation=dilation)
        self.conv1x1_residual = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.conv1x1_skip = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.gated_activation = GatedActivationUnit()
        self.residual_proj = nn.Linear(in_channels, out_channels)


    def forward(self, seq):
        """
        Apply a residual block.
        
        Args:
        * seq: a FloatTensor variable of shape (batch, in_channels, seq_length).
        
        Returns: (residual_out, skip_out) where:
        * residual_out: residual block output sequence.
        * skip_out: the skip-connection output sequence.
        """
        # 1) pass through dilated convolutions:
        conv_tanh_out = self.conv_tanh(seq)
        conv_sigmoid_out = self.conv_sigmoid(seq)
        
        # 2) pass through gated activation:
        activation_out = self.gated_activation(conv_tanh_out, conv_sigmoid_out)

        # 3) conv1x1 passes:
        conv1x1_residual_out = self.conv1x1_residual(activation_out)
        skip_out = self.conv1x1_skip(activation_out)

        # 4) final residual addition:
        reshaped_seq, axes = reshape_in(seq)
        residual_proj_out = reshape_out(self.residual_proj(reshaped_seq), axes)
        residual_out = conv1x1_residual_out + residual_proj_out

        # 5) return residual-added sequence and skip-connection sequence:
        return (residual_out, skip_out)


### ===== ===== ===== ===== Gated Activation Unit.
class GatedActivationUnit(nn.Module):
    """
    Stateless module implementing a gated activation of two
    sequences of same temporal dimension.

    Returns Tanh(x) .* Sigmoid(y)
    """
    def forward(self, x, y):
        return torch.mul(F.tanh(x), F.sigmoid(y))

    def __repr__(self):
        return self.__class__.__name__ + ' ()'
