"""
Assorted functional tools.
"""
import torch

def one_hot_encoding(seq, num_indices):
    """
    Perform one-hot encoding on sequence tensor of shape (batch, len(seq)).
    Output is of shape (batch, num_indices, len(seq)).

    Args:
    * seq: a Torch LongTensor of shape (batch, len(seq)). [Note: NOT a variable.]
    * num_indices: a python int.
    """
    return torch.zeros(seq.size(0), num_indices, seq.size(1)).scatter_(1, seq.unsqueeze(1), 1.)
