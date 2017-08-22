"""
Tests for Residual Block.
"""
import torch
from torch.autograd import Variable

from modules.block import ResidualBlock

### parameters:
in_channels = 4
out_channels = 5
sequence_length = 12
kernel_width = 2
dilation = 2
batch_size = 3

### construct input sequence:
in_seq = Variable(torch.randn(batch_size, in_channels, sequence_length))

### construct residual block and fetch outputs:
residual_block = ResidualBlock(in_channels, out_channels, kernel_width, dilation)
out_seq, skip_connection = residual_block(in_seq)

### assertions and prints:
print(out_seq.size())
print(skip_connection.size())

assert (torch.Size([batch_size, out_channels, sequence_length]) == out_seq.size())
assert (torch.Size([batch_size, out_channels, sequence_length]) == skip_connection.size())
