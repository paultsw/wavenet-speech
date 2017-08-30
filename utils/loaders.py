"""
Data-loading class.
"""
import h5py
import torch
import torch.utils.data as data
import torch.multiprocessing as mp
import numpy as np
from random import randint

# ===== ===== HDF5 Loader class ===== =====
class Loader(object):
    """
    Signal and base-sequence loader. Takes a list of datasets in the form of an HDF5 and swaps between them.

    At each call of fetch(), this loader does the following:
    1) randomly chooses a bucket;
    2) randomly chooses a pair of signals and sequences in that bucket and one-hots the signal;
    3) returns them.
    """
    def __init__(self, dataset_path, num_signal_levels=256, max_iters=100, num_epochs=1):
        """Open handle to HDF5 dataset file."""
        self.dataset_path = dataset_path
        self._dataset = h5py.File(dataset_path, 'r')
        self.num_epochs = num_epochs
        self.max_iters = max_iters
        self._load_metadata()
        self.counter = 0
        self.epochs = 0
        self.num_signal_levels = num_signal_levels
        self.cuda = False


    def cuda(self):
        self.cuda = True


    def close(self):
        """Close the dataset."""
        self._dataset.close()


    def fetch(self, bucket=None, entry=None):
        """
        Fetch a random signal/sequence pair from the dataset.
        
        Performs the following steps:
        1) choose a random bucket if bucket == None (else if int, choose that bucket);
        2) choose a random (signal,sequence) pair if entry == None (else if int, choose that entry);
        3) convert to one-hot encoding;
        4) convert to torch tensor format:
             signal ~ torch.FloatTensor variable of shape (1, num_levels, len(signal))
             sequence ~ torch.LongTensor variable of shape (1, len(sequence))
        5) if self.cuda == True, then convert to CUDA variables; return pair of variables.
        """
        ### check if we've hit the maximum number of timesteps; if yes, raise StopIteration
        self._maybe_stop()

        ### choose a random bucket and a random entry:
        bucket_id = randint(0, self._num_buckets-1) if (bucket == None) else bucket
        data_id = randint(0, self._buckets_data[bucket_id]['dataset_size']-1) if (entry == None) else entry
        
        ### fetch signal, sequence:
        signal = self.one_hot_signal(
            torch.from_numpy(self._dataset['bucket_{}'.format(bucket_id)]['signals']['{}'.format(data_id)][:]).long(),
            num_levels=self.num_signal_levels)
        seq = torch.from_numpy(
            self._dataset['bucket_{}'.format(bucket_id)]['reads']['{}'.format(data_id)][:]).long().unsqueeze(0)
        
        ### update timestep
        self._tick()

        ### return CUDA or CPU:
        if not self.cuda: return (torch.autograd.Variable(signal), torch.autograd.Variable(seq))
        return (torch.autograd.Variable(signal.cuda()), torch.autograd.Variable(seq.cuda()))


    ### Static Helper Functions:
    @staticmethod
    def one_hot_signal(signal, num_levels):
        """
        One-hot-encode a signal.
        
        Args:
        * signal: a list of ndarrays; these are the signals themselves.

        Returns:
        A one-hot-encoded signal.
        """
        return torch.zeros(1, num_levels, signal.size(0)).scatter_(1, signal.unsqueeze(0).unsqueeze(0), 1.)


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



# ===== ===== Queue-based Multiprocessing HDF5 Loader class ===== =====
class QueueLoader(object):
    """
    The QueueLoader reads from an HDF5 file and uses multiple cpu-based worker processes
    to continuously push new (signal,sequence) pairs onto the queue; the workers perform a
    one-hot encoding prior to pushing them onto the queue.

    A torch.multiprocessing.Queue object is exposed, but the individual worker processess
    are not exposed.
    
    Upon dequeuing, a (signal, sequence) pair is either returned as-is (for CPU processing)
    or is dispatched to CUDA via `*.cuda()` during the dequeue operation.


    TODO:
    * figure out how to run N worker processes in the background and kill after calls to close()
    """
    def __init__(self, dataset_path, num_signal_levels=256, num_workers=1, queue_size=50):
        """
        Construct a QueueLoader.
        """
        # save settings:
        self.dataset_path = dataset_path
        self.num_signal_levels = num_signal_levels
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.cuda = False

        # open HDF5 file:
        try:
            self.dataset = h5py.File(dataset_path, 'r')
        except e:
            raise Exception("Could not open HDF5 file: {}".format(e))

        # construct queue and initialize workers:
        self.queue = mp.Queue(queue_size)
        self.workers = []
        for k in range(num_workers):
            # [TODO: spawn worker processes here; check that this is the correct syntax]
            self.workers.append(mp.Process(target=self.queue_worker_fn, args=(self.queue,self.dataset)))

        # start all workers:
        for worker in self.workers: worker.start()


    def dequeue(self):
        """
        Dequeue a new (signal, sequence) pair from the queue as a Variable.
        """
        signal, sequence = self.queue.get()
        if self.cuda: return (torch.autograd.Variable(signal.cuda()),
                              torch.autograd.Variable(sequence.cuda()))
        return (torch.autograd.Variable(signal), torch.autograd.Variable(sequence))

    
    def close(self):
        """
        Close the queue and end the processes gracefully.
        """
        # first close the queue:
        self.queue.close()

        # kill the processes:
        for worker in self.workers: worker.join()

        # finally close HDF5 dataset file:
        self.dataset.close()


    def cuda(self):
        self.cuda = True


    def cpu(self):
        self.cuda = False


    ### Helper Functions:
    @staticmethod
    def queue_worker_fn(queue, dataset):
        """
        Read a random (signal, sequence) pair from the database, convert to LongTensor, apply a
        one-hot encoding to the signal, and append the (signal, sequence) pair to the queue.
        """
        # TODO: define worker process that performs the above, with additional checks to
        # ensure that the queue is open, not full, etc.
        if queue.full(): pass

        signal = None # [TODO: fetch from a random bucket in `dataset`]
        sequence = None # [TODO: fetch from  `dataset`]
        one_hot_signal = self.one_hot_signal(signal, self.num_signal_levels)
        queue.put( (one_hot_signal, sequence) )


    @staticmethod
    def one_hot_signal(signal, num_levels):
        """
        One-hot-encode a signal.
        
        Args:
        * signal: a list of ndarrays; these are the signals themselves.

        Returns:
        A one-hot-encoded signal.
        """
        return torch.zeros(1, num_levels, signal.size(0)).scatter_(1, signal.unsqueeze(0).unsqueeze(0), 1.)
