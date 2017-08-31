"""
A classifier network module to be stacked on top of the WaveNet.
"""
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch.autograd import Variable

from modules.block import ResidualBlock
from modules.conv_ops import reshape_in, reshape_out

class WaveNetClassifier(nn.Module):
    """
    Specification of a classifier network; used to turn a WaveNet into a sequence classifier, e.g. for Speech-to-Text.
    """
    def __init__(self, in_dim, num_labels, layers, out_dim, pool_kernel_size=2,
                 input_kernel_size=2, input_dilation=1, softmax=True):
        """
        Constructor for WaveNetClassifier.

        Args:
        * in_dim: python int; the dimensionality of the input sequence.
        * num_labels: the number of labels in the softmax distribution at the output.
        * layers: list of (non-causal) convolutional layers to stack. Each entry is of the form
          (in_channels, out_channels, kernel_size, dilation).
        * out_dim: the final dimension of the output before the dimensionality reduction to logits over labels.
        * pool_kernel_size: python int denoting the receptive field for mean-pooling. This determines downsample rate.
        * input_kernel_size: size of the internal kernel of the conv1d going from input to conv-stack.
        * input_dilation: dilation for conv block going from input to conv-stack.
        * softmax: if True, softmax the output layer before returning. If False, return un-normalized sequence.
        """
        ### parent constructor
        super(WaveNetClassifier, self).__init__()

        ### attributes
        self.in_dim = in_dim
        self.num_labels = num_labels
        self.layers = layers
        self.num_layers = len(layers)
        self.out_dim = out_dim
        # mean pooling:
        self.pool_kernel_size = pool_kernel_size
        self.pool_padding = 0
        # input 1x1Conv layer:
        self.input_kernel_size = input_kernel_size
        self.input_padding = _autopad(input_kernel_size, input_dilation)
        self.input_dilation = input_dilation
        # convolutional stack:
        self.layers = layers
        # output layer
        self.out_kernel_size = out_kernel_size
        self.out_dilation = out_dilation
        self.softmax = softmax

        ### submodules
        # mean pooling layer:
        self.mean_pool = nn.AvgPool1d(kernel_size=pool_kernel_size, padding=self.pool_padding)

        # input layer:
        self.input_block = ResidualBlock(in_dim, layers[0][0], input_kernel_size, input_dilation,
                                         causal=False)
        self.input_skip_bottleneck = nn.Conv1d(layers[0][0], kernel_size=1, padding=0, dilation=1)

        # stack of residual convolutions and their bottlenecks for skip connections:
        convolutions = []
        skip_conn_bottlenecks = []
        for (c_in,c_out,k,d) in layers:
            convolutions.append( ResidualBlock(c_in, c_out, k, d, causal=False) )
            skip_conn_bottlenecks.append( nn.Conv1d(c_out, out_dim, kernel_size=1, padding=0, dilation=1) )
        self.convolutions = nn.ModuleList(stack)
        self.bottlenecks = nn.ModuleList(skip_conn_bottlenecks)

        # (1x1 Conv + ReLU + 1x1 Conv) stack, going from output dimension to logits over labels:
        self.output_block = nn.Sequential(
            nn.LeakyReLU(0.01),
            nn.Conv1d(out_dim, out_dim, kernel_size=1, dilation=1),
            nn.LeakyReLU(0.01),
            nn.Conv1d(out_dim, num_labels, kernel_size=1, dilation=1))

        ### sensible initializations for parameters:
        for p in self.input_block.parameters():
            if len(p.size()) > 1: nn_init.kaiming_uniform(p)
            if len(p.size()) == 1: p.data.zero_()
        for p in self.convolution_stack.parameters():
            if len(p.size()) > 1: nn_init.kaiming_uniform(p)
            if len(p.size()) == 1: p.data.zero_()
        for p in self.bottlenecks.parameters():
            if len(p.size()) == 2: nn_init.eye(p)
            if len(p.size()) == 1: p.data.zero_()


    def forward(self, seq):
        """
        Run the sequence classification stack on an input sequence.
        
        Args:
        * seq: a FloatTensor variable of shape (batch_size, in_seq_dim, seq_length).
        
        Returns:
        * logit_seq: a FloatTensor variable of shape (batch_size, out_seq_dim, seq_length).
        """
        # 1. initial mean-pooling to down-sample:
        out = self.mean_pool(seq)
        
        # pass thru the input layer block:
        skips_sum = Variable(seq.data.new(seq.size(0), self.out_dim, signal.size(2)).fill_(0.))
        out, skip = self.input_block(mean_pool_seq)
        skips_sum = skips_sum + self.input_skip_bottleneck(skip)

        # run through convolutional stack (& accumulate skip connections thru bottlenecks):
        for l in range(self.num_layers):
            out, skip = self.convolutions[l](out)
            skips_sum = skips_sum + self.bottlenecks[l](skip)

        # run through output stack:
        logit_seq = self.output_block(skips_sum)

        if not self.softmax: return logit_seq
        
        reshaped_logit_seq, _axes = reshape_in(logit_seq)
        return reshape_out(F.softmax(reshaped_logit_seq), _axes)
