"""
ByteNet decoder, implemented using only Linear layers (with torch.gather operations used to
implement dilational effects).

All modules in this file operate on (Seq, Batch, Dim) tensors with the exception of the core
LinearByteNetDecoder module, which performs a reshape from (Batch, Dim, Seq) => (Seq, Batch, Dim)
in the forward() method. This latter operation is done to preserve dimensionality with the
encoder, which is based on convolutions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#===== Modules =====
### LayerNorm, specialized for our use-case here:
class LayerNorm(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

### Linearized multiplicative unit:
class MultiplicativeUnit(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

### residual block that uses multiplicative units:
class ResidualMUBlock(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

### Linearized ByteNet-style decoder:
class LinearByteNetDecoder(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
