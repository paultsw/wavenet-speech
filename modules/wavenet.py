"""
Description of WaveNet module.
"""
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch.autograd import Variable

from modules.block import ResidualBlock
from modules.conv_ops import CausalConv1d, reshape_in, reshape_out

class WaveNet(nn.Module):
    """
    The WaveNet module is a stack of residual causal convolutions with dilations which
    describes a generative model of signals.

    Contains the following modules (order as they appear in the stack from bottom-up):
    1) Causal Conv1d
    2) Stacked Residual Blocks
    3) Output Stack on Skip-Connections

    Recommended settings:
    * in_dim given by dataset
    * entry_kwidth = 2
    * layers = [(in_dim, in_dim, 2, d**2 % 512) for d in range(num_layers)]
    * out_dim = in_dim
    """
    def __init__(self, in_dim, entry_kwidth, layers, out_dim, softmax=True):
        """
        Construct a new WaveNet module with a given set of layers.

        Args:
        * in_dim: dimension of inputs.
        * entry_kwidth: kernel size for the bottom-most causal conv1d layer.
        * layers: a list of length `num_layers` where each entry is of the form (c_in, c_out, kernel_width, dilation),
          used to describe a residual convolutional block.
        * out_dim: dimension of outputs.
        * softmax: if True, apply a softmax as the final output; if False, return un-normalized outputs.
        """
        ###
        super(WaveNet, self).__init__()
        
        ### save inputs as attributes:
        self.in_dim = in_dim
        self.entry_kwidth = entry_kwidth
        self.layers = layers
        self.num_layers = len(layers)
        self.out_dim = out_dim
        self.softmax = softmax

        ### construct all modules in the WaveNet stack:
        # initial undilated causal conv going into the network:
        self.entry_conv1d = CausalConv1d(in_dim, layers[0][0], entry_kwidth, dilation=1)

        # a stack of residual blocks forms the core of the network; also
        # create bottlenecks connecting skip connections to output:
        conv_stack = []
        bottlenecks = []
        for (c_in, c_out, kwidth, dilation) in layers:
            conv_stack.append( ResidualBlock(c_in, c_out, kwidth, dilation) )
            bottlenecks.append(nn.Conv1d(c_out, out_dim, 1, padding=0, dilation=1))
        self.convolutions = nn.ModuleList(conv_stack)
        self.bottlenecks = nn.ModuleList(bottlenecks)

        # output stack: 1x1 Conv => ReLU => 1x1 Conv
        self.output_stack = nn.Sequential(
            nn.LeakyReLU(0.01),
            nn.Conv1d(out_dim, out_dim, kernel_size=1, padding=0, dilation=1),
            nn.LeakyReLU(0.01),
            nn.Conv1d(out_dim, out_dim, kernel_size=1, padding=0, dilation=1))

        ### initialize all inputs:
        for p in self.entry_conv1d.parameters():
            if len(p.size()) > 1: nn_init.kaiming_uniform(p)
            if len(p.size()) == 1: p.data.zero_()
        for p in self.convolutions.parameters():
            if len(p.size()) > 1: nn_init.kaiming_uniform(p)
            if len(p.size()) == 1: p.data.zero_()
        for p in self.bottlenecks.parameters():
            if len(p.size()) == 2: nn_init.eye(p)
            if len(p.size()) == 1: p.data.zero_()
        for p in self.output_stack.parameters():
            if len(p.size()) > 1: nn_init.kaiming_uniform(p)
            if len(p.size()) == 1: p.data.zero_()


    def forward(self, signal):
        """
        Forward-propagate a signal through all of the layers of the conv stack.
        """
        # 1) compute initial causal conv1d:
        out = self.entry_conv1d(signal)

        # 2) pass through all layers of the stack of residual blocks:
        # create zeros with dtype like `signal`:
        skips_sum = Variable(signal.data.new(signal.size(0), self.out_dim, signal.size(2)).fill_(0.))
        for l in range(self.num_layers):
            out, skip = self.convolutions[l](out)
            skips_sum = skips_sum + self.bottlenecks[l](skip)
        
        # 3) process the collection of skip-connections:
        output_seq = self.output_stack(skips_sum)
        
        if not self.softmax: return output_seq

        # 4) optional final softmax on output:
        output_seq_reshape, axes = reshape_in(output_seq)
        predicted_signal = reshape_out(F.softmax(output_seq_reshape), axes)

        return predicted_signal
