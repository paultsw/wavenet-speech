"""
Helper functions that may be useful in deciding upon a model architecture.
"""
def compute_receptive_field(dilation_depth, nb_stacks):
    """
    This function is modified/borrowed from:
    https://github.com/basveeling/wavenet/blob/master/wavenet.py#L280

    We implicitly assume a doubling of dilations at each step, i.e. each "stack"
    is of the form `dilations := [1, 2, 4, ..., ]`.
    
    Args:
    * dilation_depth: this is a python integer representing `len(dilations)-1` with
    `dilations` as defined above.
    * nb_stacks: the number of times the `dilations` block is repeated.
    """
    return nb_stacks * (2 ** dilation_depth * 2) - (nb_stacks - 1)
