"""
Unit tests for logit sequence => label sequence decoders.
"""
import torch
from torch.autograd import Variable

from modules.sequence_decoders import argmax_decode, labels2strings, BeamSearchDecoder


##### construct example data:
sequence_length = 19
batch_size = 3
num_labels = 5

# labels in dense and one-hot-encoded formats:
labels = torch.rand(batch_size, sequence_length).mul_(num_labels).long()
labels_one_hot = torch.zeros(batch_size, sequence_length, num_labels).scatter_(2, labels.unsqueeze(2), 1.)

# create un-normalized logits; add very small amount of noise:
logits = labels_one_hot + torch.randn(labels_one_hot.size()).mul_(0.1)

##### test argmax decode:
decoded_labels = argmax_decode(logits)
print("Decoded labels:")
print(decoded_labels)
print("Original labels:")
print(labels)

##### test stringify:
print(labels2strings(labels, lookup={0: 'A', 1: 'G', 2: 'C', 3: 'T', 4: ''}))


##### test beam search decoder:
BeamSearchDecoder(beam_width=4)(logits)
