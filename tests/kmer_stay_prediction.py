"""
Attempt to predict kmers and STAY symbols using the RawCTCNet encoder.

This training test proceeds as follows:
1) Generate random 5mer sequences: `K := (k1, k2, k3, ..., kN)`.
2) Create a random alignment (based on gamma-distributed duration model) by adding `STAY` characters:
   `k_stay := (k1, STAY, STAY, k2, STAY, STAY, STAY, k3, k4, STAY, ..., kN, STAY, STAY)`.
3) Convert `k_stay` to signals using a 5mer model (NPZ format).
4) attempt to predict `k_stay` using a RawCTCNet with 
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from modules.raw_ctcnet import RawCTCNet
import numpy as np


### Stay model data generator:
class StayGenerator(object):
    """
    Wrapper class for stay-sequence generator.
    
    Primary functionality is the `fetch()` method, which returns two sequences: `kmer_stay` and
    `signal`. The latter is the input sequence and the former is the target sequence.
    """
    def __init__(self, kmer_model_npz_path, duration_rate, duration_shape):
        """Constructor; store relevant data in the object."""
        self.duration_rate = duration_rate
        self.duration_shape = duration_shape
        self.npz_path = kmer_model_npz_path
        npz_file = np.load(kmer_model_npz_path)
        self.kmer_model_means = npz_file['means']
        self.kmer_model_stdvs = npz_file['stdvs']
        

    def fetch(self):
        """Generate a new (input,target) sequence-pair on the fly."""
        pass


### main loop: generate model with fixed configuration and train against artificial data:
def main():
    """Run training test."""
    # hyperparameters:
    num_labels = 1025 # == 4**5 + 1
    out_dim = 1025
    num_features = 1025
    layers = [(256, 256, 2, d) for d in [1,2,4,8,16]] * 5
    # model:
    ctcnet = RawCTCNet(num_features, feature_kwidth, num_labels, layers, out_dim,
                       positions=True, softmax=False, causal=False)
