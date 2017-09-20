"""
Recipes for loading data from heterogeneous HDF5 dataset and placing them onto a queue.

Each function here must take an HDF5 filehandle and extract data sequences, placing
them onto a multiprocessing Queue in torch.{Float|Int|Long}Tensor format.
"""
import h5py
import torch
import random
import numpy as np

##### Worker function for e.coli datasets:
def ecoli_worker_fn(hdf5_handle, keys, batch_size=8, sample_lengths=(90,110), num_levels=256):
    """
    Sample random data sequences from hdf5 file, pad signals to equal sizes (as LongTensor),
    one-hot the signals, and returns the resulting batched torch.Tensors.
    """
    # select a random read:
    read = random.choice(keys)
    quantized = hdf5_handle[read]['quantized']
    positions = hdf5_handle[read]['raw']['positions']
    references = hdf5_handle[read]['reference']

    # sample `batch_size` subintervals of `positions` with lengths within `sample_lengths`:
    _lens = np.random.randint(sample_lengths[0], sample_lengths[1], size=batch_size)
    _starts = np.random.randint(0, (positions.shape[0]-sample_lengths[1]), size=batch_size)
    _stops = _starts+_lens
    subintervals = [(_starts[k], _stops[k]) for k in range(batch_size)]

    # extract corresponding reference sequences:
    base_seqs = []
    base_lengths_list = []
    for start_ix, stop_ix in subintervals:
        move_to_start = positions[start_ix] - positions[0]
        move_to_stop = positions[stop_ix] - positions[0]
        ref_subseq = references[move_to_start:move_to_stop]
        base_seqs.append(ref_subseq)
        base_lengths_list.append(ref_subseq.shape[0])

    # extract corresponding quantized samples and one-hot encode:
    signals = []
    signal_lengths_list = []
    for start_ix, stop_ix in subintervals:
        subsignal = np.concatenate(quantized[start_ix:stop_ix])
        signal_length = subsignal.shape[0]
        one_hot_signal = np.zeros((num_levels, signal_length), dtype=np.float32)
        one_hot_signal[ subsignal, np.arange(signal_length) ] = 1. # fancy indexing
        signals.append(one_hot_signal)
        signal_lengths_list.append(signal_length)

    # bundle all data in an appropriate format:
    flattened_seqs = np.concatenate(base_seqs)
    base_lengths = np.array(base_lengths_list, dtype=np.int32)
    signal_lengths = np.array(signal_lengths_list, dtype=np.int32)
    batch_sigs = _batchify_signals(signals, np.amax(signal_lengths))

    # enqueue batched data:
    signals_th = torch.from_numpy(batch_sigs)
    seqs_th = torch.from_numpy(flattened_seqs)
    signal_lengths_th = torch.from_numpy(signal_lengths)
    base_lengths_th = torch.from_numpy(base_lengths)
    return (signals_th, seqs_th, signal_lengths_th, base_lengths_th)


#----- E.Coli Worker Fn Helpers:
def _batchify_signals(signals_list, max_length):
    """
    Pad a list of np.float32 arrays with 0-vectors, where each array is of shape (num_levels, seq_length_L).
    """
    padded = []
    for sig in signals_list:
        padding_size = max_length - sig.shape[1]
        padded.append( np.pad(sig, ((0,0),(0,padding_size)), mode='constant') )
    return np.stack(padded, axis=0)


##### Worker function for bucketed datasets:
def bucket_worker_fn():
    """
    Sample from a bucketed dataset. (TBD)
    """
    pass
