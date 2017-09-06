"""
Load data from a glob of NPZs.
"""
import numpy as np
import os
from glob import glob
import torch
import torch.utils.data as data
import torch.multiprocessing as mp
import random


class FilenameDataset(data.Dataset):
    """
    A dataset consisting of filenames.
    """
    def __init__(self, files_glob, num_epochs=1):
        """
        Construct a ...
        """
        self._files = random.shuffle(glob(files_glob))
        self._num_files = len(self._files)
        self._num_epochs = num_epochs


    def __len__(self):
        """
        Get length of the dataset; number of files in the filename dataset.
        """
        return (self._num_files * num_epochs)


    def __getitem__(self):
        """
        Fetch another element of the dataset.
        """
        return random.choice(self._files)


class NPZQueueLoader(object):
    """
    Load dense data sequence pairs from a glob of NPZ files. The files are lazily loaded,
    i.e. they are batched (and padded/one-hot-encoded, in the case of signals) and placed onto
    the queue at the moment

    Each item on the queue is a 4-ple (signals, sequences, signal_lengths, sequence_lengths)
    where:

    * signals: [cuda.]FloatTensor variable of shape (batch_size, num_signal_levels, max_length)
    * sequences: IntTensor variable of shape ( sum([len(seq) for seq in batch]), );
    these are the flattened target sequences. Note that they must be flattened for CTC loss.
    * signal_lengths: an IntTensor giving the lengths of all the signals. Shape (batch_size,)
    * sequence_lengths: an IntTensor giving the lengths of all the sequences. Shape (batch_size,)
    """
    def __init__(self, files_glob, num_signal_levels, batch_size, num_epochs, num_workers, queue_size):
        """
        Construct a loader.
        """
        # input args:
        self.files_glob = files_glob
        self.num_signal_levels = num_signal_levels
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.queue_size = queue_size
        
        # meta properties:
        self._cuda = False
        
        # internal objects:
        self.filename_dataset = FilenameDataset(files_glob, num_epochs)
        self.filename_loader = data.DataLoader(self.filename_dataset, batch_size=1, shuffle=True)
        self.queue = mp.Queue(queue_size)
        self.workers = [mp.Process(target=self.queue_worker_fn, args=(self.queue, self.filename_loader))]
        
        # start workers:
        for worker in self.workers: worker.start()

    def cuda(self):
        self._cuda = True

    def cpu(self):
        self.cuda = False
        
    def close(self):
        for worker in self.workers: worker.join()
        self.queue.close()

    @staticmethod
    def queue_worker_fn(queue, filename_loader):
        """TBD"""
        pass

    @staticmethod
    def one_hot_encode(seqs):
        """TBD"""
        pass

    @staticmethod
    def pad_batch(seq_of_seqs):
        """pad a list of lists to a uniform sequence length. [TODO]"""
        batched_seq = None # batch of padded sequences
        seq_lengths = None # signal lengths on each batch
        return (batched_seq, seq_lengths)

    @staticmethod
    def flatten_sequences(seq_of_seqs):
        """Flatten a list of sequences and return along with sequence lengths. [TODO]"""
        flattened_seq = None # flattened list
        seq_lengths = None   # length of each sequence in the batch
        return (flattened_seq, seq_lengths)
