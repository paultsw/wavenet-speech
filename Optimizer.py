import torch.optim as optim
from torch.optim import lr_scheduler as lr_tools
from torch.autograd import Variable

class Optimizer(object):
    """Container class for optimizers."""
    def __init__(self, parameters, optim_type, lr, weight_decay=None, reduce_lr=False):
        """
        Construct an optimizer.

        `parameters` is expected to be either the output of some nn.Module.parameters() call
        or a list of dictionaries of the form
          [ {"params": nn.Module.parameters()} ]

        N.B. (TODO): `weight_decay` and `reduce_lr` options currently unsupported.
        """
        assert optim_type in ['adam', 'adagrad']
        self.optim_type = optim_type
        self.weight_decay = weight_decay
        self.reduce_lr = reduce_lr

        if optim_type == 'adam':
            self.optimizer = optim.Adam(parameters, lr=lr)
        if optim_type == 'adagrad':
            self.optimizer = optim.Adagrad(parameters, lr=lr)

    def adjust_lr(self, new_lr=None):
        """
        Adjust the learning rate of the optimizer.
        """
        pass # [TODO: support this in the future]

    def step(self):
        """Take an optimization step."""
        self.optimizer.step()

    def zero_grad(self):
        """Zero-out the gradients from the previous step."""
        self.optimizer.zero_grad()
