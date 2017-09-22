"""
Data-loading classes.
"""
import h5py
import torch
import torch.utils.data as data
import threading as thread
import queue
import numpy as np
from random import randint, shuffle

# ===== ===== Queue-based Multiprocessing HDF5 Loader class ===== =====
from utils.worker_fns import ecoli_worker_fn, bucket_worker_fn
class QueueLoader(object):
    """
    The QueueLoader reads from an HDF5 file and uses multiple cpu-based worker processes
    to continuously push new (signal,sequence) batches onto the queue; the workers perform
    (sampling => one-hot encoding => padding) prior to pushing them onto the queue.

    A Queue object is exposed, but the individual worker threads are not exposed.
    
    Upon dequeuing, a tuple of batched sequences is either returned as-is (for CPU processing)
    or is dispatched to CUDA via `*.cuda()` during the dequeue operation; the tuple is
    of the form ().
    """
    def __init__(self, dataset_path, num_signal_levels=256, num_workers=1, queue_size=50, batch_size=8, sample_lengths=(90,110),
                 max_iters=100, epoch_size=1000, worker_fn='ecoli'):
        """
        Construct a QueueLoader.
        """
        # save settings:
        self.dataset_path = dataset_path
        self.num_signal_levels = num_signal_levels
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.sample_lengths = sample_lengths
        self.max_iters = max_iters
        self.epoch_size = epoch_size
        self.num_epochs = 0
        self._cuda = False

        # open HDF5 file:
        try:
            self.dataset = h5py.File(dataset_path, 'r')
        except:
            raise Exception("Could not open HDF5 file: {}".format(dataset_path))

        # list of top-level read names into train/validation set:
        self.reads = list(self.dataset.keys())
        valid_set_size = int(0.3 * len(self.reads)) # 30% of reads go to validation
        self.train_reads = self.reads[valid_set_size:]
        self.valid_reads = self.reads[0:valid_set_size]

        # construct queue, re-entrant lock, and data counter:
        self.train_queue = queue.Queue(queue_size)
        self.valid_queue = queue.Queue(queue_size)
        self.queue_lock = thread.Lock()
        self.global_counter = 0
        self.stop_event = thread.Event()

        # create worker function as a closure:
        assert (worker_fn in ['ecoli', 'buckets'])
        if worker_fn == 'buckets': raise Exception("ERR: bucketed dataset is currently unsupported.")
        def queue_worker(q, reads, stop_evt):
            while True:
                if stop_evt.is_set(): break
                if (self.global_counter > self.max_iters): break
                datavals = ecoli_worker_fn(self.dataset, reads, batch_size=batch_size, sample_lengths=sample_lengths)
                with self.queue_lock:
                    q.put( datavals )
                    self.incr_global_ctr()
        self.queue_worker = queue_worker

        # create workers:
        self.workers = []
        for k in range(num_workers):
            _train_worker = thread.Thread(target=self.queue_worker,
                                          args=(self.train_queue, self.train_reads, self.stop_event),
                                          daemon=True)
            _train_worker.start()
            self.workers.append(_train_worker)
        # only need one validation worker since we dequeue from it so rarely:
        _valid_worker = thread.Thread(target=self.queue_worker,
                                      args=(self.valid_queue, self.valid_reads, self.stop_event),
                                      daemon=True)
        _valid_worker.start()
        self.workers.append(_valid_worker)

    def incr_global_ctr(self):
        self.global_counter += 1
        if (self.global_counter % self.epoch_size == 0): self.num_epochs += 1

    def dequeue(self, from_queue="train"):
        """
        Dequeue a new data sequence tuple from the queue as a list of Variables.
        """
        assert (from_queue in ['train', 'valid'])
        if from_queue == "train":
            try:
                vals = self.train_queue.get(timeout=1)
            except:
                raise StopIteration
        if from_queue == "valid":
            try:
                vals = self.valid_queue.get(timeout=1)
            except:
                raise StopIteration
        if self._cuda:
            return [torch.autograd.Variable(val.cuda()) for val in vals]
        else:
            return [torch.autograd.Variable(val) for val in vals]
    
    def close(self):
        """
        Gracefully join all threads and close HDF5 file handle.
        """
        # join the processes, with timeout of 2 seconds:
        self.stop_event.set()
        [worker.join(timeout=2) for worker in self.workers]

        # finally close HDF5 dataset file:
        if self.dataset: self.dataset.close()

    def cuda(self):
        self._cuda = True

    def cpu(self):
        self._cuda = False


# ===== ===== Basic HDF5 Loader class ===== =====
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
        self._closed = False
        self.num_epochs = num_epochs
        self.max_iters = max_iters
        self._load_metadata()
        self.counter = 0
        self.epochs = 0
        self.num_signal_levels = num_signal_levels
        self._cuda = False


    def cuda(self):
        self._cuda = True


    def close(self):
        """Close the dataset."""
        self._dataset.close()
        self._closed = True


    def is_closed(self):
        """True if closed, False if opened."""
        return self._closed


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
        if not self._cuda: return (torch.autograd.Variable(signal), torch.autograd.Variable(seq))
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
            raise StopIteration
