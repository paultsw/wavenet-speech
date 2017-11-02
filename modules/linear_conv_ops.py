"""
Linearized convolutional modules.

These are convolutional modules which additionally offer a `linear()` method, which
provides a functional interface to the underlying parameters of the convolutional
kernel; this allows e.g. convolutions to be computed *incrementally* across the sequence,
as we require in the decoder.

Additionally, there is a property `*.receptive_field` that determines the required timesteps
before a call of `linear()` can occur; this property increases in the sequence dimension as
modules containing a LinearConv1d submodule get more complex, and can be used as a check to
ensure that enough timesteps are available.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

### Linearized analogue of Conv1d
class LinearConv1d(nn.Conv1d):
    """
    Same interface/usage as nn.Conv1d, but adds optional an optional `linear()` method.
    Calls to `linear()` uses the parameters of the Conv1d as the weights to a linear
    feedforward layer. Calls to `forward()` loop the `linear()` function over as many
    frames of the input sequence as a standard nn.Conv1d would.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        # run parent constructor with same parameters:
        super(LinearConv1d,self).__init__(in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, groups, bias)
        # internal value that determines which steps to keep per frame, taking dilation into account:
        self._ker_ixs = get_ker_ixs(dilation, kernel_size)

    def forward(self, in_seq):
        """Loop over calls to `linear()` method; returns the same output as `nn.Conv1d.forward()`."""
        start_ix = None # [CALCULATE FROM PAD/STRIDE]
        stop_ix = None # [CALCULATE FROM PAD/STRIDE]

    def linear(self, frame, keep_dims=False):
        """
        Args:
        * frame: a {cuda.,cpu.}FloatTensor Variable of shape (batch, in_channels, receptive_field);
        this is the frame at a given timestep that will be functionally passed through a linear layer
        given by the convolutional kernel and bias.
        
        Returns: a {cuda.,cpu.}FloatTensor Variable of shape (batch, out_channels) given by passing
        the input through the convolutional layer.
        
        If `keep_dims == True`, the output is of shape `(batch, out_channels, 1)`.
        """
        # sanity check:
        assert ( frame.size(2) == self.receptive_field )

        # run forward pre hooks (e.g., weight norm)
        for hook in self._forward_pre_hooks.values():
            hook(self, frame)

        # compute `out ~ (N, c_out)` using reshaped kernel and bias:
        out = F.linear(
            # # reshape frame to (N, c_in * ker_size), dropping the dilation-skipped timesteps:
            frame[:,:,self._ker_ixs].view(frame.size(0), frame.size(1)*self.kernel_size[0]),
            # reshape kernel: (c_out, c_in, ker_size) => (c_out, c_in * ker_size)
            self.weight.view(self.weight.size(0), self.weight.size(1) * self.weight.size(2)),
            bias=self.bias)

        # return, possibly with dummy dimension:
        out = out.unsqueeze(2) if keep_dims else out
        return out

    @property
    def receptive_field(self):
        """Determines the number of timesteps needed before we can output a single frame in `linear()`."""
        return ( self.kernel_size[0] + (self.dilation[0]-1)*(self.kernel_size[0]-1) )

# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Helper Functions
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
def compute_new_length(seq_len, pad, dil, ker):
    """Compute the output length of a Conv1d given input sequence length, padding amount,
    dilation rate, and kernel width."""
    return torch.floor(torch.FloatTensor([seq_len + 2*pad - (dil * (ker-1))]))[0]


def reshape_in(seq):
    """Reshapes an (N, C, L) tensor to (N*L, C). This is useful for preparing inputs to Linear or Softmax layers."""
    N, C, L = seq.size()
    return seq.permute(0,2,1).contiguous().view(N*L,C), (N,L)


def reshape_out(seq, dims):
    """Reshapes an (N*L, C) tensor to (N, C, L) with assistance from a `dims` parameter of the form dims == (N,L).
    This is useful as an inverse operation to reshape_in, to be applied after a nonlinearity that requires (N,C) inputs."""
    N, L = dims
    return seq.view(N, L, -1).permute(0,2,1).contiguous()


def autopad(k,d):
    """
    Given dilation and kernel width, automatically calculate correct amount to pad on left+right
    needed to preserve temporal dimensionality.

    [N.B.: this fails for even kernel and odd dilation values, for unknown reason...]
    """
    total_padding = (k-1) * d
    # if total padding is odd:
    if (total_padding % 2 == 1):
        return int((total_padding-1) / 2)+1#, int((total_padding-1) / 2)
    else:
        return int(total_padding / 2)#, int(total_padding / 2)


def get_ker_ixs(d,k):
    """
    Given a kernel size and a dilation rate, this function tells you which frame timesteps
    to keep when performing a call to `linear()`.
    
    N.B.: this should only be used in the implementation of LinearConv1d; for modules that
    employ LinearConv1d as a submodule, you should lean on the correctness of the implementation
    of `LinearConv1d.linear()` when constructing the module's `linear()` method.
    """
    total_frame_size = k*d - (d-1)
    keep_ixs = [i for i in range(total_frame_size) if (i % d == 0)]
    return keep_ixs
