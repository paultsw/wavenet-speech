"""
Attempt to decode a set of kmers into a nucleotide sequence using the ByteNet decoder.

Results: [pending]
"""
import numpy as np
from scipy.ndimage.filters import generic_filter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from modules.bytenet_decoder import ByteNetDecoder

### Construct function to convert random nucleotide sequences to kmer labels:
batch_size = 32 # [TODO: implement batching]
nt_length = 100
nts_to_5mer_map = lambda nt: np.sum(nt-np.ones(nts.shape) * np.array([2**4, 2**3, 2**2, 2**1, 2**0]))
def fetch_stitch_nts():
    # generate a random nucleotide sequence:
    nts = np.random.randint(1,high=5,size=nt_length, dtype=np.int32)
    # stitch into 5mers:
    kmers = generic_filter(nts, nts_to_5mer_map, size=(5,), mode='constant')
    kmers = kmers[4:-4].astype(int) # remove padding values
    return torch.from_numpy(nts).long(), torch.from_numpy(kmers).long()


### Construct bytenet decoder:
decoder = ByteNetDecoder(None) # [TODO: parameters]


### construct loss & optimizer:
loss_fn = nn.CrossEntropyLoss()
opt = optim.AdaMax(decoder.parameters(), lr=0.002)

### training loop:
num_iterations = 2000000 # 2M
print_every = 50
try:
    best_observed = 1000000.
    for k in range(num_iterations):
        opt.zero_grad()
        # generate data/prediction:
        nts, kmers = fetch_stitched_nts()
        pred_nts = decoder.train(kmers)
        # compute loss:
        loss = 0.
        for k in range(pred_nts.size(0)):
            loss = loss + loss_fn(pred_nts[k], nts[k])
        # backprop:
        loss.backward()
        opt.step()
        # log if scheduled:
        if (k % print_every == 0):
            loss_scalar = loss.data[0]
            avg_loss_scalar = loss_scalar / pred_nts.size(0)
            if avg_loss_scalar < best_observed: best_observed = avg_loss_scalar
            print("Step {0} | Loss: {1} | Avg Loss: {2}".format(k, loss_scalar, avg_loss))
except KeyboardInterrupt:
    print("Halted training from keyboard.")
finally:
    print("Best observed loss: {}".format(best_observed))
