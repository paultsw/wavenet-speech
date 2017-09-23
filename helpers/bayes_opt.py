"""
Bayesian optimization helpers for WaveNet-CTC model.

TODO: integrate with Spearmint or Hyperband for bayesian optimization.
"""
import torch

class BayesOpt(object):
    """
    Takes a series of dicts with config settings and runs model against those configs
    for some number of fixed timesteps. Returns statistics about the best config.

    Relies on gaussian process regression to choose continuous hyperparameters;
    discrete hyperparameters (e.g. turning early stopping on/off) need to be tried by
    running different configurations.
    """
    pass
