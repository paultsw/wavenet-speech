"""
Recipes for loading data from heterogeneous HDF5 dataset and placing them onto a queue.
"""
import h5py
import torch
import random

##### Worker function for e.coli datasets:
def ecoli_worker_fn(hdf5_handle, keys, queue, batch_size=8, sample_lengths=(90,110), num_levels=256):
    """
    Sample random data sequences from hdf5 file, pad signals to equal sizes (as LongTensor),
    one-hot the signals, and the resulting batched torch.Tensors onto the queue.

    TODO: figure out a parallelizable/efficient way of subsampling?
    """
    # TODO: perform all sanity checks/validation steps here:
    if queue.full(): pass

    # select a random read:
    read = random.choice(keys)
    quantized = hdf5_handle[read]['quantized']
    positions = hdf5_handle[read]['raw']['positions']
    references = hdf5_handle[read]['reference']

    base_seqs = []
    signals = []
    for k in range(batch_size):
        # fetch random subsequence from positions:
        subsample = _subsample(positions, random.randint(sample_lengths[0], sample_lengths[1]))
        
        # fetch associated reference sequences:
        sequence = None # [TODO]

        # extract a collection of random signals from quantized dataset:
        signal = None # [TODO: fetch from a random bucket in `dataset`]

        base_seqs.append(sequence)
        signals.append(signal)

    # batch everything together:
    batch_seqs = torch.stack(base_seqs, dim=0).long()
    batch_sigs = torch.stack(signals, dim=0)

    # perform one-hot encoding on the signals:
    one_hot_signal = _one_hot_encode(signal, num_levels)

    # place batches onto queue:
    queue.put( (one_hot_signals, batch_seqs) )


##### Worker function for bucketed datasets:
def bucket_worker_fn():
    """
    Sample from a bucketed dataset. (TBD)
    """
    pass


##### Helper functions:
def _subsample_signal(signal_ds, length):
    """Sample a sub-signal of a given length from a larger signal."""
    pass # [TBD]


def _one_hot_encode(tsr, nlevels):
    """Perform one-hot encoding to convert a 2D LongTensor into a 3D one-hot-encoded FloatTensor."""
    pass # [TBD]


def _reference_correspondence(pos_seq):
    """
    Given a sequence of positions, compute the corresponding reference subsequence by first
    computing the moves.
    """
    pass # [TBD]
