"""
Convolutional modules.
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
    def __init__(self, in_channels, out_channels, kernel_width, dilation=1):
        """Create underlying causal convolution."""
        # run parent initialization:
        super(CausalConv1d, self).__init__()
        
        # save arguments as attributes:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_width
        self.dilation = dilation
        # amount to pad/shift by; ensures that `self.padding` elements left at end,
        # which we can safely ignore when computing forward pass
        self.padding = (kernel_width-1)*dilation

        # single conv1d as submodule:
        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_width, stride=1,
            padding=self.padding,
            dilation=dilation)


    def forward(self, seq):
        """
        Forward pass through the internal conv1d and removes the last `self.padding` elements.
        """
        conv1d_out = self.conv1d(seq)
        return conv1d_out[:,:,0:seq.size(2)]


class NonCausalConv1d(nn.Module):
    """
    An automatically-padded Conv1d.
    """
    def __init__(self, in_channels, out_channels, kernel_width, dilation=1):
        """Create an auto-padded conv1d."""
        # run parent initialization:
        super(NonCausalConv1d, self).__init__()

        # save arguments as attributes:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_width
        self.dilation = dilation
        # amount to pad/shift by:
        self.padding = autopad(kernel_width, dilation)

        # single conv1d as submodule:
        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_width, stride=1,
            padding=self.padding,
            dilation=dilation)


    def forward(self, seq):
        """
        Forward pass through the internal conv1d; slice off the same dimension as the original
        sequence length to preserve temporal resolution.
        """
        return self.conv1d(seq)[:,:,0:seq.size(2)]


# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Helper Functions
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
def compute_new_length(seq_len, pad, dil, ker):
    """Compute the output length of a Conv1d given input sequence length, padding amount,
    dilation rate, and kernel width."""
    return torch.floor(torch.FloatTensor([seq_len + 2*pad - (dil * (ker-1))]))[0]


def reshape_in(seq):
    """Reshapes an (N, C, L) tensor to (N*L, C). This is useful for preparing inputs to Linear or Softmax layers."""
    N, C, L = seq.size()
    return seq.permute(0,2,1).contiguous().view(N*L,C), (N,L)


def reshape_out(seq, dims):
    """Reshapes an (N*L, C) tensor to (N, C, L) with assistance from a `dims` parameter of the form dims == (N,L).
    This is useful as an inverse operation to reshape_in, to be applied after a nonlinearity that requires (N,C) inputs."""
    N, L = dims
    return seq.view(N, L, -1).permute(0,2,1).contiguous()


def autopad(k,d):
    """
    Given dilation and kernel width, automatically calculate correct amount to pad on left+right
    needed to preserve temporal dimensionality.

    [N.B.: this fails for even kernel and odd dilation values, for unknown reason...]
    """
    total_padding = (k-1) * d
    # if total padding is odd:
    if (total_padding % 2 == 1):
        return int((total_padding-1) / 2)+1, int((total_padding-1) / 2)
    else:
        return int(total_padding / 2), int(total_padding / 2)
