"""
A classifier network module to be stacked on top of the WaveNet.
"""
import torch
import torch.nn as nn
from torch.autograd import Variable


class WaveNetClassifer(nn.Module):
    """
    Specification of a classifer network; used to turn a WaveNet into a sequence classifer.
    """
    def __init__(self, num_layers):
        """
        Constructor for WaveNetClassifier.

        Args:
        * num_layers: number of (non-causal) convolutional layers to stack.
        """
        ### parent constructor
        super(WaveNetClassifier, self).__init__()

        ### attributes
        self.num_layers = num_layers

        ### submodules
        # TODO: Figure out arguments for each of the following:
        _modules = [(
            'meanpool', nn.AvgPool1d(kernel_size=None, stride=None, padding=0, None, None)
        )]
        for k in range(num_layers):
            _modules.append((
                'conv{}'.format(l), nn.Conv1d(None,None,None,None)
            ))
        self.classifier_stack = nn.Sequential(OrderedDict(_modules))

    def forward(self, seq):
        """Run the sequence classification stack on an input sequence."""
        return self.classifier_stack(seq)
