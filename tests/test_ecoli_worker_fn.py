"""
Validate ecoli worker function's sampling methods.
"""
import torch
import numpy as np
import h5py
from utils.worker_fns import ecoli_worker_fn

data = h5py.File("data/ecoli/reads.01.reference.hdf5", 'r')
reads = list(data.keys())

### extract a sample (use `debug_mode` to get special debugging-specific outputs)
readname, pos_interval, ref_seq, one_hot_samples = ecoli_worker_fn(data, reads, batch_size=1, sample_lengths=(90,110),
                                                                   num_levels=256, debug_mode=True)

### compare with the data on the HDF5 file:
# readname :: String
# pos_interval :: [ (Int,Int) ]
# ref_seq :: Torch.IntTensor ~ [ Int ]
# one_hot_samples :: Torch.LongTensor ~ (1,256,LEN)
#=================================================
print("*** Positions:")
read_gp = data[readname]
pos0 = pos_interval[0][0]
pos1 = pos_interval[0][1]
positions = read_gp['raw']['positions'][pos0:pos1]
print(positions)

#=================================================
print("*** Signal data:")
print("Sampled:")
_, vals = torch.max(one_hot_samples[0], dim=0)
dense_vals = (list(vals))
print(dense_vals)
print("Num samples: {}".format(len(dense_vals)))
print("Quantized:")
merged_quantized = np.concatenate([xx for xx in read_gp['quantized'][pos0:pos1]], axis=0)
print(merged_quantized)
print("Num samples: {}".format(len(merged_quantized)))

#=================================================
### Lookup helper functions
# Taken from this stackoverflow answer: https://stackoverflow.com/a/7100681/3141064
def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a. strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

list_ref_seq = list(ref_seq)
print("*** Reference sequences:")
print("Sampled:")
print(list_ref_seq)
print("Looking for sampled subsequence on HDF5 references:")
boolarr = rolling_window(read_gp['reference'][:], len(list_ref_seq)) == list_ref_seq
for k in range(len(boolarr)):
    num_trues = 0
    for ll in boolarr[k]:
        if ll: num_trues += 1
    found = (float(num_trues) / float(len(boolarr[k])) > 0.8)
    if found: print("Found subseq at {}".format(k))
    break
print("Uh oh, subseq not found")
