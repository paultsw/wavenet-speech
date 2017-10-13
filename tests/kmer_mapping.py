"""
Attempt to learn the gaussian 5mer model.

Results: the below model fails to reproduce the kmer model in the NPZ file after 2M loops.
This seems to indicate that network depth and temporal dependencies are very
important for reproducing the kmer-to-gaussian mapping.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

### Construct data generator that converts 
kmer_model_path = "../../utils/r9.4_450bps.5mer.template.npz"
kmer_model_npz = np.load(kmer_model_path)
kmer_means = kmer_model_npz['means']
kmer_stdvs = kmer_model_npz['stdvs']
batch_size = 32
def kmer_to_gaussian():
    # generate random kmers:
    kmer_ixs = np.random.randint(0,high=1024, size=batch_size, dtype=np.int32)
    # look up means/stdvs:
    means = np.array([kmer_means[k] for k in kmer_ixs])
    stdvs = np.array([kmer_stdvs[k] for k in kmer_ixs])
    # generate samples:
    samples = np.random.normal(loc=means, scale=stdvs)
    # convert to TH and return:
    kmers_th = torch.from_numpy(kmer_ixs).long()
    samples_th = torch.from_numpy(samples).float().unsqueeze(1)
    return (Variable(kmers_th),
            Variable(samples_th))

### Construct featurization layer:
print("Building network and initializing weights...")
nhid = 2048
feature_layer = nn.Sequential(nn.Linear(1,1),
                              nn.Linear(1, nhid),
                              nn.LeakyReLU(nn.init.calculate_gain('leaky_relu')),
                              nn.Linear(nhid, nhid),
                              nn.LeakyReLU(nn.init.calculate_gain('leaky_relu')),
                              nn.Linear(nhid, nhid),
                              nn.LeakyReLU(nn.init.calculate_gain('leaky_relu')),
                              nn.Linear(nhid, nhid),
                              nn.LeakyReLU(nn.init.calculate_gain('leaky_relu')),
                              nn.Linear(nhid, nhid),
                              nn.LeakyReLU(nn.init.calculate_gain('leaky_relu')),
                              nn.Linear(nhid, nhid),
                              nn.LeakyReLU(nn.init.calculate_gain('leaky_relu')),
                              nn.Linear(nhid,1024))
for p in feature_layer.parameters():
    if len(p.size()) > 1: nn.init.sparse(p, sparsity=0.1)
    if len(p.size()) == 1: p.data.zero_().add_(0.0001)
print("...Done.")

### Build loss and optimizer:
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adamax(feature_layer.parameters(), lr=0.002)

### Training loop:
num_iterations = 100000000
print_every = 100
try:
    print("Training...")
    for k in range(num_iterations):
        opt.zero_grad()
        kmers, samples = kmer_to_gaussian()
        logits = feature_layer(samples)
        loss = loss_fn(logits, kmers)
        if k % print_every == 0: print("Step {0} | Loss {1}".format(k, loss.data[0]))
        opt.step()
except KeyboardInterrupt:
    print("...Training interrupted.")
finally:
    print("...Training finished.")
