"""
The decoder from ByteNet, as described in:

"Neural Machine Translation in Linear Time", N. Kalchbrenner et al,
https://arxiv.org/abs/1610.10099
"""

import torch
import torch.nn as nn
from modules.block import ResidualReLUBlock, ResidualMultiplicativeBlock

class ByteNetDecoder(nn.Module):
    """
    The ByteNet decoder module is a stack of causal dilated conv1d blocks that takes a batch of
    logits at each timestep, mixes it with an encoded representation, and returns a batch of logits
    representing the predicted next timestep.

    In training mode, we can take the whole target sequence as input and train against the same target
    sequence after shifting by 1; however, in evaluation mode we must loop around the convolutional
    module by linearizing it and feeding the previous timestep's prediction back into it.
    """
    def __init__(self, num_labels, channels, encoding_dim, kwidth, output_dim, layers, block='multiplicative'):
        """
        Construct all submodules and save parameters.
        
        Args:
        * num_labels: the size of the alphabet (including the NULL/<BLANK> CTC character).
        * channels: the number of input channels; each tensor in the network will either have
        `channels` or `2*channels` dimensions (depending on the specific sub-module.)
        * encoding_dim: dimension of the output timesteps of the encoded source sequence.
        * kwidth: the kernel size for the stack of residual blocks.
        * output_dim: the dimensionality of the output mapping layers.
        * layers: a python list of integer tuples of the form [(kwidth, dilation)].
        * block: either 'mult' or 'relu'; decides which type of causal ResConv block to use.
        """
        super(ByteNetDecoder, self).__init__()
        # save inputs:
        self.num_labels = num_labels
        self.channels = channels
        self.encoding_dim = encoding_dim
        self.output_dim = output_dim
        self.layers = layers
        if not (block in []):
            raise TypeError("ARGUMENT: `block` must be either 'relu' or 'mult'.")
        self.block = block
        ResBlock = ResidualMultiplicativeBlock if (block == 'mult') else ResidualReLUBlock
        
        # construct input embedding and Conv1x1 layer:
        self.input_layer = nn.Sequential(
            nn.Embedding(num_labels, 2*channels)
            self.input_conv1x1 = nn.Conv1d(2*channels, 2*channels, kernel_size=1, stride=1, dilation=1))
        
        # conv1x1 to mix in the encoded sequence:
        self.encoding_layer = nn.Conv1x1(encoding_dim, 2*channels, kernel_size=1, stride=1, dilation=1)
        
        # stack of causal residual convolutional blocks:
        self.stacked_residual_layer = nn.Sequential(OrderedDict(
            [('resconv{}'.format(l),ResBlock(2*channels, k, dilation=d)) for (l,(k,d)) in enumerate(layers)]
        ))
        
        # Final [Conv1x1=>ReLU=>Conv1x1] mapping to construct outputs:
        self.output_layer = nn.Sequential(
            nn.Conv1d(2*channels, output_dim, kernel_size=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(output_dim, num_labels, kernel_size=1, dilation=1))

        # initialize parameters:
        self.init_params()


    def init_params(self):
        """Good initial values for all tensor/matrix parameters."""
        for p in self.parameters():
            if len(p.size()) >= 2: nn.init.kaiming_normal(p)
            if len(p.size()) == 1: p.data.zero_().add(0.0001 * torch.randn(p.size()))


    def forward(self, target_seq, encoded_seq):
        """
        Forward pass for training. (To perform inference without a known target sequence, use `evaluate()`.)
        
        Expects `target_seq` to be a sequence of type Variable(LongTensor) and `encoded_seq` to be a
        sequence of type Variable(FloatTensor).
        
        N.B.: You need to shift the `target_seq` value manually in the training loop! This is important;
        if you don't shift the target sequence, this entire module will just learn to be a very expensive
        version of the identity mapping.
        """
        out_seq = self.input_layer(target_seq)
        out_seq = out_seq + self.encoding_layer(encoded_seq)
        out_seq = self.stacked_residual_layer(out_seq)
        out_seq = self.output_layer(out_seq)
        return out_seq


    def evaluate(self, x0, encoded_seq):
        """
        Given an initial timestep, perform evaluation using a linearized version of the modules.
        
        Expects an initial timestep input x0 ~ LongTensor vector.
        """
        # [TBD: what's a good way of linearizing a stack of convolutions in torch???]
        # [To save time, work on this function /after/ noticeable gains in training.]
        return None
