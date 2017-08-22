"""
Tests for WaveNet module.
"""
import torch
from torch.autograd import Variable

from modules.wavenet import WaveNet

### parameters
kwidth = 2 # constant kernel width across all convolutional layers
batch_size = 5
seq_length = 14
seq_dim = 11 # for both input and output, since we're feeding one-hot and outputting softmax distributions
dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
layers = [(seq_dim, seq_dim, kwidth, d) for d in dilations]

### input sequence:
input_seq = Variable(torch.randn(batch_size, seq_dim, seq_length))

### construct WaveNet:
wavenet = WaveNet(seq_dim, kwidth, layers, seq_dim)

### run wavenet on inputs:
predictions = wavenet(input_seq)

### prints, asserts, etc:
#print(predictions.size())
print(predictions)
assert (torch.Size([batch_size, seq_dim, seq_length]) == predictions.size())

