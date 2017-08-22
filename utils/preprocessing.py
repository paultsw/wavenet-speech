"""
Data pre-processing ops.
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def quantize_signal(signal_tensor, num_classes):
    """
    Take a batch of sequences and quantize it into a set of levels. This essentially discretizes a signal.
    """
    pass
