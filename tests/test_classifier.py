"""
Tests for WaveNet-Speech convolutional transcription network.
"""
import torch
from torch.autograd import Variable

from modules.classifier import WaveNetClassifier

### constructor parameters:
num_labels = 8
batch_size = 3
seq_dim = 256
seq_length = 10000
dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
layers = [(seq_dim, seq_dim, 2, d) for d in dilations]
downsample_rate = 3

### construct fake data:
input_seq = Variable(torch.randn(batch_size, seq_dim, seq_length))

### construct wavenet classifier/transcriptor stack & run:
classifier = WaveNetClassifier(seq_dim, num_labels, layers, pool_kernel_size=downsample_rate)
out_dist_seq = classifier(input_seq)

### print outputs, sizes, etc:
print("Downsample rate:")
print(downsample_rate)
print("Input Sequence Size:")
print(input_seq.size())
print("Output Sequence Size:")
print(out_dist_seq.size())
#print(out_dist_seq)

assert (batch_size == out_dist_seq.size(0))
assert (num_labels == out_dist_seq.size(1))