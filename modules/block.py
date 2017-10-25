"""
Definition of a skip-connection residual block, the base component of
the WaveNet architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.conv_ops import  CausalConv1d, NonCausalConv1d, reshape_in, reshape_out
from modules.layernorm import LayerNorm
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
        AutoConv1d = CausalConv1d if causal else NonCausalConv1d
        self.conv_tanh = AutoConv1d(in_channels, out_channels, kernel_width, dilation=dilation)
        self.conv_sigmoid = AutoConv1d(in_channels, out_channels, kernel_width, dilation=dilation)
        self.conv1x1_residual = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.conv1x1_skip = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.gated_activation = GatedActivationUnit()
        self.residual_proj = nn.Linear(in_channels, out_channels)

        # compute receptive field:
        self.receptive_field = self.conv_tanh.receptive_field


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


### ===== ===== ===== ===== Residual Multiplicative Block.
class ResidualMUBlock(nn.Module):
    """
    A residual multiplicative block (causal) as specified in the ByteNet architecture:
    ("Neural Machine Translation in Linear Time", Kalchbrenner et al, https://arxiv.org/abs/1610.10099).
    """
    def __init__(self, nchannels, k_width, dilation=1):
        """Construct a residual multiplicative block; create all submodules."""
        super(ResidualMUBlock, self).__init__()
        # store inputs:
        self.nchannels = nchannels
        self.k_width = k_width
        self.dilation = dilation
        half_channels = int(nchannels/2)

        # submodules:
        self.stack = nn.Sequential(
            LayerNorm(nchannels),
            nn.ReLU(),
            # Conv1x1, dimensionality reduction by 1/2:
            nn.Conv1d(nchannels, half_channels, 1, stride=1, dilation=1),
            LayerNorm(half_channels, dim=1),
            nn.ReLU(),
            # 1xK MU:
            MultiplicativeUnit(half_channels, k_width, dilation=dilation),
            # 1x1 MU:
            MultiplicativeUnit(half_channels, 1, dilation=1),
            # Conv1x1, dimensionality expansion to 2x:
            nn.Conv1d(half_channels, nchannels, 1, stride=1, dilation=1))

        # compute receptive field:
        self.receptive_field = self.stack[5].receptive_field

    def forward(self, seq):
        """Pass through the [LayerNorm => Conv => MultUnit] stack and add to residual."""
        return (seq + self.stack(seq))

    def init(self):
        """Initialize parameters via kaiming-normal on tensors and noisy-zero on biases."""
        for p in self.parameters():
            if len(p.size()) >= 2: nn.init.kaiming_normal(p)
            if len(p.size()) == 1: p.data.zero_().add(0.001 * torch.randn(p.size()))


### ===== ===== ===== ===== Residual ReLU Block.
class ResidualReLUBlock(nn.Module):
    """
    A ByteNet residual block that uses ReLUs instead of Multiplicative units, as described in ByteNet.
    (Note that the authors use ReLU activations for machine translation and MUs for language modelling.)

    Note: this block expects inputs of even dimension, as there is a half-downsampling operation inside
    the block's main module.
    """
    def __init__(self, nchannels, k_width, dilation=1):
        """Construct a residual ReLU block; create all submodules."""
        super(ResidualReLUBlock, self).__init__()
        # store inputs:
        self.nchannels = nchannels
        self.k_width = k_width
        self.dilation = dilation
        half_channels = int(nchannels/2)
        
        # submodules:
        self.stack = nn.Sequential(
            LayerNorm(nchannels),
            nn.ReLU(),
            # Conv1x1, dimensionality reduction by 1/2:
            nn.Conv1d(nchannels, half_channels, 1, stride=1, dilation=1),
            LayerNorm(half_channels),
            nn.ReLU(),
            # Causal Conv1xK:
            CausalConv1d(half_channels, half_channels, kernel_width=k_width, dilation=dilation),
            LayerNorm(half_channels),
            nn.ReLU(),
            # Conv1x1, increase dimension 2x:
            nn.Conv1d(half_channels, nchannels, 1, stride=1, dilation=1))

        # compute receptive field:
        self.receptive_field = self.stack[5].receptive_field

    def forward(self, seq):
        """Pass through the [LayerNorm => Conv => ReLU] stack and add to residual."""
        return (seq + self.stack(seq))

    def init(self):
        """Initialize parameters via kaiming-normal on tensors and noisy-zero on biases."""
        for p in self.parameters():
            if len(p.size()) >= 2: nn.init.kaiming_normal(p)
            if len(p.size()) == 1: p.data.zero_().add(0.001 * torch.randn(p.size()))


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


### ===== ===== ===== ===== Multiplicative Unit module.
class MultiplicativeUnit(nn.Module):
    """
    A multiplicative "activation"; similar to the version in the Video Pixel Network architecture:
    "Video Pixel Networks" (sec. 4.1),Kalchbrenner et al, https://arxiv.org/abs/1610.00527

    N.B.:
    * this module is *NOT* stateless; it contains four Conv1ds as submodules.
    * This is *NOT* exactly the same as the version in the paper; as it is used in the ByteNet decoder,
    we use causal convolutions here.
    """
    def __init__(self, ndim, k, dilation=1):
        super(MultiplicativeUnit, self).__init__()
        self.ndim = ndim
        self.gate1 = CausalConv1d(ndim, ndim, kernel_width=k, dilation=dilation)
        self.gate2 = CausalConv1d(ndim, ndim, kernel_width=k, dilation=dilation)
        self.gate3 = CausalConv1d(ndim, ndim, kernel_width=k, dilation=dilation)
        self.update = CausalConv1d(ndim, ndim, kernel_width=k, dilation=dilation)
        self.init() # initialize parameters

        self.receptive_field = max([self.gate1.receptive_field, self.gate2.receptive_field,
                                    self.gate3.receptive_field, self.update.receptive_field])

    def forward(self, h):
        g1 = F.sigmoid(self.gate1(h))
        g2 = F.sigmoid(self.gate2(h))
        g3 = F.sigmoid(self.gate3(h))
        u = F.tanh(self.update(h))
        # g1 * tanh( g2 * h + g3 * u ):
        return (g1.mul(F.tanh(g2.mul(h) + g3.mul(u))))

    def init(self):
        for p in self.parameters():
            if len(p.size()) >= 2: nn.init.kaiming_normal(p)
            if len(p.size()) == 1: p.data.zero_().add(0.001 * torch.randn(p.size()))
