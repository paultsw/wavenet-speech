"""
Helper functions for training.
"""
#import torch


def calculate_receptive_field(dilations, kernel_width):
    """
    Takes a list of dilations and a kernel width (the uniform kernel size of each layer) and computes
    the size of the receptive field, i.e. the number of timesteps in the source sequence that each
    output timestep can observe.

    Can be used for both causal and standard conv1ds (the equation is the same).
    """
    # base case:
    if len(dilations) == 1:
        d = dilations[0]
        k = kernel_width
        return (k * (k-1) * (d-1))
    # recursion:
    else:
        calculate_receptive_field(dilations[1:], kernel_width)
        # [TODO: figure out recursive relationship here]


def compute_receptive_field(dilation_depth, nb_stacks):
    """
    This function is modified/borrowed from:
    https://github.com/basveeling/wavenet/blob/master/wavenet.py#L280
    """
    return nb_stacks * (2 ** dilation_depth * 2) - (nb_stacks - 1)
