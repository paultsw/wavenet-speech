"""
A classifier network module to be stacked on top of the WaveNet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.conv_ops import reshape_in, reshape_out

class WaveNetClassifier(nn.Module):
    """
    Specification of a classifier network; used to turn a WaveNet into a sequence classifier, e.g. for Speech-to-Text.
    """
    def __init__(self, in_dim, num_labels, layers,
                 pool_kernel_size=2, input_kernel_size=2, input_dilation=1,
                 out_kernel_size=2, out_dilation=1, softmax=True):
        """
        Constructor for WaveNetClassifier.

        Args:
        * in_dim: python int; the dimensionality of the input sequence.
        * num_labels: the number of labels in the softmax distribution at the output.
        * layers: list of (non-causal) convolutional layers to stack. Each entry is of the form
          (in_channels, out_channels, kernel_size, dilation).
        * pool_kernel_size: python int denoting the receptive field for mean-pooling. This determines downsample rate.
        * input_kernel_size: size of the internal kernel of the conv1d going from input to conv-stack.
        * input_dilation: dilation for conv1d going from input to conv-stack.
        * out_kernel_size: python int denoting the size of the kernel for the output conv layer.
        * out_dilation: python int denoting the amount of dilation to use for the output conv layer's kernel.
        * softmax: if True, softmax the output layer before returning. If False, return un-normalized sequence.
        """
        ### parent constructor
        super(WaveNetClassifier, self).__init__()

        ### attributes
        self.in_dim = in_dim
        self.num_labels = num_labels
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
        self.out_padding = _autopad(out_kernel_size, out_dilation)
        self.out_dilation = out_dilation
        self.softmax = softmax

        ### submodules
        # mean pooling layer:
        self.mean_pool = nn.AvgPool1d(kernel_size=pool_kernel_size, padding=self.pool_padding)

        # input layer:
        input_conv1d = nn.Conv1d(in_dim, layers[0][0], input_kernel_size, padding=self.input_padding,
                                 dilation=input_dilation)

        # conv1d stack:
        stack = [nn.Conv1d(c_in, c_out, k, padding=_autopad(k,d), dilation=d) for (c_in,c_out,k,d) in layers]
        self.convolution_stack = nn.Sequential(input_conv1d, *stack)

        # output layer:
        self.output_conv1d = nn.Conv1d(layers[-1][1], num_labels, out_kernel_size, padding=self.out_padding,
                                       dilation=out_dilation)


    def forward(self, seq):
        """
        Run the sequence classification stack on an input sequence.
        
        Args:
        * seq: a FloatTensor variable of shape (batch_size, in_seq_dim, in_seq_length).
        
        Returns:
        * predictions: a FloatTensor variable of shape (batch_size, out_seq_dim, out_seq_length).
        
        (In general, it is hard to predict the output sequence length if the padding/dilation rates are not
        chosen to specifically preserve temporal dimensionality.)
        """
        mean_pool_seq = self.mean_pool(seq)
        conv_stack_out = self.convolution_stack(mean_pool_seq)
        output_seq = self.output_conv1d(conv_stack_out)
        
        if not self.softmax: return output_seq
        
        reshaped_output_seq, _axes = reshape_in(output_seq)
        return reshape_out(F.softmax(reshaped_output_seq), _axes)


# ===== ===== ===== ===== Helper Functions: automatically calculate correct padding amounts
from math import floor, ceil
def _autopad(k,d):
    """
    Given dilation and kernel width, automatically calculate correct amount to pad on left+right
    needed to preserve temporal dimensionality.
    """
    pad_times_2 = (k-1) * d
    if (pad_times_2 % 2 == 1):
        return int(floor(pad_times_2 / 2)), int(ceil(pad_times_2 / 2))
    else:
        return int(pad_times_2 / 2), int(pad_times_2 / 2)
