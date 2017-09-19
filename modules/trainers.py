"""
Wrappers on torch.nn loss functions and optimizers.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from warpctc_pytorch import CTCLoss


class CustomOptimizer(object):
    """
    Wrapper for several optimizers and schedulers.

    List of Optimizers:
    * Adam
    * SGD(+Nesterov)
    * RMSProp
    """
    def __init__(self, optim_type, **kwargs):
        """
        Construct a new optimizer object from scratch.

        `optim_type` is one of 'adam', 'sgd', 'rmsprop', ...
        """
        pass

    def call()


class JointLoss(object):
    """
    Construct a custom loss function with weighting on alphabet.
    
    Combines a CTC loss and a cross entropy loss on a tuple of
    input sequences and a tuple of output sequences.
    """
    def __init__(self, joint_weight=0.5, ce_weights=None, batch_average=False, timestep_average=False):
        """
        Returns a JointLoss function.

        The parameter `joint_weight` is the sliding weight that lets you choose
        whether CTC loss or cross-entropy loss is more important; at `weight := 0.0`,
        this is equivalent to solely minimizing the cross entropy loss; at `weight := 1.0`,
        this is equivalent to solely minimizing the CTC loss.
        """
        self.joint_weight = weight
        self.timestep_average = timestep_average
        self.ce_weights = ce_weights
        self.ctc_loss_fn = CTCLoss()
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=ce_weights, size_average=batch_average)


    def __call__(self, pred_samples, next_samples,
                 pred_logits, true_labels,
                 logit_sizes, label_sizes):
        """
        Calls the joint-loss function.
        """
        pass # [TBD]
