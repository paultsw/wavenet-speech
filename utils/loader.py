"""
Data-loading class.
"""
import h5py
import torch
import torch.utils.data as data
import numpy as np
import random

# ===== ===== HDF5 Loader class ===== =====
class Loader(object):
    """
    Signal and base-sequence loader. Takes a list of datasets in the form of an HDF5 and swaps between them.

    At each call of fetch(), this loader does the following:
    1) randomly chooses a bucket;
    2) randomly chooses a subset of bucket.dataset_size of cardinality `batch_size`;
    3) loads the corresponding batch of signals and sequences.
    """
    def __init__(self, dataset_path, batch_size=1, max_iters=100, num_epochs=1):
        """Open handle to HDF5 dataset file."""
        self.dataset_path = dataset_path
        self._dataset = h5py.File(dataset_path, 'r')
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_iters = max_iters
        self._load_metadata()
        self.counter = 0
        self.epochs = 0


    def close(self):
        """Close the dataset."""
        self._dataset.close()


    def fetch(self):
        """
        Fetch a random batch from the dataset.
        
        Performs the following steps:
        1) choose a random bucket;
        2) choose a random assortment of (signal,sequence) pairs from bucket in batch format ~ (batch, in-dim, seq-len);
        3) perform padding;
        4) convert to torch tensor format;
        5) return batch.
        """
        ### check if we've hit the maximum number of timesteps; if yes, raise StopIteration
        self._maybe_stop()

        ### main operations: choose a random bucket and a random set of sequences:
        bucket = self.random_bucket(self._num_buckets)
        data_ids = self.random_sequences(self._buckets_data[bucket]['dataset_size'], self.batch_size)
        return self.fetch_specific_batch(bucket, data_ids)
        
        ### update timestep
        self._tick()


    def fetch_batch(self, bucket_id, data_nums):
        """
        Fetch a specific batch.

        Args:
        * bucket_id: choice of the bucket to fetch from.
        * data_nums: list of sequence IDs (possibly in np.array([int]) format).
        """
        pass # [TODO: FIX THIS]


    def fetch_one(self, bucket_id, data_num):
        """
        Fetch a specific (signal,sequence) pair.
        """
        return (self._dataset['bucket_{}'.format(bucket_id)]['signals']['{}'.format(data_num)][:],
                self._dataset['bucket_{}'.format(bucket_id)]['reads']['{}'.format(data_num)][:])
        
    ### Static Helper Functions:
    @staticmethod
    def random_bucket(num_buckets):
        """Choose a bucket at random."""
        return random.random(0,num_buckets-1)

    @staticmethod
    def random_sequences(num_sequences, batch_size):
        """Choose a batch at random. Randomly chooses an integer list of length `batch_size` from [0,num_sequences)."""
        return np.random.choice(num_sequences, size=batch_size, replace=False)

    ### Helper Methods:
    def _load_metadata(self):
        """Load metadata values as attributes."""
        self._num_buckets = self._dataset['meta'].attrs['num_buckets']
        self._bucket_size = self._dataset['meta'].attrs['bucket_size']
        self._signals_path = self._dataset['meta'].attrs['signals_path']
        self._reads_path = self._dataset['meta'].attrs['reads_path']
        self._lengths_tsv_path = self._dataset['meta'].attrs['lengths_tsv_path']
        self._buckets_data = {}
        for k in range(self._num_buckets):
            self._buckets_data[k] = {
                'dataset_size': self._dataset['bucket_{}'.format(k)].attrs['dataset_size'],
                'max_read_length': self._dataset['bucket_{}'.format(k)].attrs['max_read_length'],
                'min_read_length': self._dataset['bucket_{}'.format(k)].attrs['min_read_length'],
                'max_signal_length': self._dataset['bucket_{}'.format(k)].attrs['max_signal_length'],
                'min_signal_length': self._dataset['bucket_{}'.format(k)].attrs['min_signal_length'],
                # n.b.: you have to index these to load them into memory; this is a space-saving
                # technique by design
                'read_lengths': self._dataset['bucket_{}'.format(k)]['read_lengths'],
                'signal_lengths': self._dataset['bucket_{}'.format(k)]['signal_lengths'],
            }
        self._num_sequences = sum([self._buckets_data[k]['dataset_size'] for k in range(self._num_buckets)])

    def _tick(self):
        """Update counter and maybe update epoch"""
        self.counter += 1
        if (self.counter != 0 and self.counter % self._num_sequences == 0): self.epochs += 1

    def _maybe_stop(self):
        """Return True if we are done; return False otherwise"""
        if self.epochs == self.num_epochs or self.counter == self.max_iters:
            self.close()
            raise StopIteration


# ===== ===== Load the same data over and over again: ===== =====
# [TODO: deprecate this once the above loader works.]
class OverfitLoader(object):
    """
    Loads a single (signal, base-sequence) pair over and over again.
    """
    def __init__(self, signal_path, sequence_path, batch_size=1, num_levels=256):
        """Construct a tensor dataset."""
        # store attributes for later reference:
        self.signal_path = signal_path
        self.sequence_path = sequence_path
        self.batch_size = batch_size
        self.num_levels = num_levels
        
        # create one-hot encoding of signal:
        signal_tensor = torch.from_numpy(np.load(signal_path)).unsqueeze(1)
        one_hot_signal_tensor = torch.zeros(signal_tensor.size(0), num_levels)
        one_hot_signal_tensor.scatter_(1, signal_tensor, 1.)

        sequence_tensor = torch.from_numpy(np.load(sequence_path))
        
        # stack the signal tensor multiple times:
        self._stacked_signal_tensor = torch.stack([one_hot_signal_tensor] * batch_size, dim=0)

        # stack the sequence tensor multiple times:
        self._stacked_sequence_tensor = torch.stack([sequence_tensor] * batch_size, dim=0)


    def fetch(self):
        """Return the same batch over and over again."""
        return (self._stacked_signal_tensor, self._stacked_sequence_tensor)
