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



# ===== ===== Queue-based Multiprocessing HDF5 Loader class ===== =====
from utils.worker_fns import ecoli_worker_fn, bucket_worker_fn
class QueueLoader(object):
    """
    The QueueLoader reads from an HDF5 file and uses multiple cpu-based worker processes
    to continuously push new (signal,sequence) batches onto the queue; the workers perform
    (sampling => one-hot encoding => padding) prior to pushing them onto the queue.

    A torch.multiprocessing.Queue object is exposed, but the individual worker processess
    are not exposed.
    
    Upon dequeuing, a (signal, sequence) batch is either returned as-is (for CPU processing)
    or is dispatched to CUDA via `*.cuda()` during the dequeue operation.

    TODO:
    * figure out how to run N worker processes in the background and kill after calls to close()
    * figure out if torch.multiprocessing.Event() is necessary to synchronize the processes. See:
      https://discuss.pytorch.org/t/tensors-as-items-in-multiprocessing-queue/411
    """
    def __init__(self, dataset_path, num_signal_levels=256, num_workers=1, queue_size=50, batch_size=8, sample_lengths=(90,110),
                 worker_fn='ecoli'):
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
        self._cuda = False

        # open HDF5 file:
        try:
            self.dataset = h5py.File(dataset_path, 'r')
        except e:
            raise Exception("Could not open HDF5 file: {}".format(e))

        # list of top-level read names:
        self.reads = list(self.dataset.keys())

        # construct queue:
        self.queue = mp.Queue(queue_size)

        # create worker function as a closure:
        # [TODO: multiplex over different worker function types here]
        assert (worker_fn in ['ecoli', 'buckets'])
        self.queue_worker_fn = lambda: ecoli_worker_fn(self.dataset, self.reads, self.queue,
                                                       batch_size=batch_size,
                                                       sample_lengths=sample_lengths)

        # create workers:
        self.workers = [mp.Process(target=self.queue_worker_fn) for k in range(num_workers)]

        # start all workers:
        for worker in self.workers: worker.start()


    def dequeue(self):
        """
        Dequeue a new (signal, sequence) pair from the queue as a Variable.
        """
        signal, sequence = self.queue.get()
        if self._cuda: return (torch.autograd.Variable(signal.cuda()),
                              torch.autograd.Variable(sequence.cuda()))
        return (torch.autograd.Variable(signal), torch.autograd.Variable(sequence))

    
    def close(self):
        """
        Close the queue and end the processes gracefully.
        """
        # first close the queue:
        self.queue.close()

        # join the processes:
        for worker in self.workers:
            worker.join()

        # finally close HDF5 dataset file:
        self.dataset.close()


    def cuda(self):
        self._cuda = True


    def cpu(self):
        self._cuda = False


# ===== ===== Pore Generator Loader class ===== =====
from scipy.ndimage.filters import generic_filter
from scipy.signal import triang

# default mapping from nucleotides to underlying pico-amp values:
_CURRS_ = { 0: 51., 1: 22., 2: 103., 3: 115. }
class PoreModelLoader(object):
    """
    An on-line random generator for artificial pore data.
    """
    def __init__(self, max_iters, num_epochs, epoch_size, batch_size=1, num_levels=256,
                 lengths=(20,30), pore_width=4, sample_rate=3, currents_dict=_CURRS_, sample_noise=3.0):
        """
        Initialize the pore loader.
        
        Data/iterator settings:
        * max_iters: maximum number of times that fetch() can be called before StopIteration is raised.
        * num_epochs: maximum number of epochs (defined by `epoch_size`) that can be ran before StopIteration.
        * epoch_size: number of examples per epoch.
        * batch_size: number of data sequences per batch. [Default: 1]
        * num_levels: number of discrete levels in the signal sequence.
        
        Model settings:
        * lengths: a tuple indicating the minimum and maximum base sequence lengths to sample from.
        * pore_width: indicates how many nucleotides can fit inside the pore at a given time.
        * sample_rate: number of samples to emit per movement of the pore.
        * currents_dict: a { nucleotide => pico-amps } lookup dict.
        * sample_noise: python float, indicates the amount of white noise to add to the signal.
        """
        # save input params:
        self.max_iters = max_iters
        self.num_epochs = num_epochs
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.num_levels = num_levels
        self.min_length = lengths[0]
        self.max_length = lengths[1]
        self.pore_width = pore_width
        self.sample_rate = sample_rate
        self.currents_dict = currents_dict
        self.sample_noise = sample_noise

        # internal record-keeping parameters:
        self.counter = 0
        self.epochs = 0
        self.on_cuda = False

        # define quantization law and levels:
        _mu = float(num_levels)
        _mu_law = lambda x: (np.sign(x) * (np.log(1+_mu*np.abs(x)) * np.reciprocal(np.log(1+_mu))))
        self.quant_law = np.vectorize(_mu_law)
        self.quant_levels = np.linspace(-1.0, 1.0, num=num_levels)


    def pore_model_fn(self, sequence):
        """
        Converts a sequence of nucleotides into a sequence of pico-amps.
        """
        pico_amps = np.array([self.currents_dict[nucleo] for nucleo in sequence], dtype=np.float32)
        triangular_window = triang(self.pore_width)
        fn = lambda arr: np.dot(arr, triangular_window)
        pA_sequence = generic_filter(pico_amps, fn, size=self.pore_width, mode='constant', cval=0.0)
        noiseless = np.repeat(pA_sequence, self.sample_rate)
        noise = np.random.normal(loc=0.0, scale=self.sample_noise, size=noiseless.shape)
        return (noiseless + noise)


    def quantize_fn(self, fseq):
        """
        Quantize a signal to some number of levels; first shift+scale the inputs to the right dimensions
        before applying quant law.
        """
        normalized = (fseq - np.mean(fseq)) / (np.amax(fseq) - np.amin(fseq))
        mapped = self.quant_law(normalized)
        return np.digitize(mapped, self.quant_levels)


    def one_hot_fn(self, dseq):
        """
        Perform one-hot encoding on a quantized sequence. Takes 1D np.int32 array as input, returns
        a one-hot encoded np.float32 array with shape (num_levels, len(dseq),) as output.

        [TBD]
        """
        seq_length = dseq.shape[0]
        ohe_arr = np.zeros((self.num_levels, seq_length), dtype=np.float32)
        ohe_arr[ dseq, np.arange(seq_length) ] = 1. # fancy indexing to do the trick
        return ohe_arr


    def convert_to_signal(self, seq):
        picoamp_signal = self.pore_model_fn(seq)
        quantized = self.quantize_fn(picoamp_signal)
        one_hot_signal = self.one_hot_fn(quantized)
        return one_hot_signal

    @staticmethod
    def batchify(signals_list):
        """
        Pad a list to uniform length with vectors consisting solely of zeros.
        
        Args:
        * signals_list: a list of np.float32 arrays, each of shape (signal_length,).
        Returns:
        * signals_batch: np.float32 of shape ( len(signals_list), num_levels, max[signal_length] ).
        """
        # get max length:
        pad_length = max([sig.shape[1] for sig in signals_list])
        # pad all ndarrs and stack together in 0-axis:
        padded_sigs = []
        for sig in signals_list:
            padded_sigs.append( np.pad(sig, ((0,0),(0,pad_length-sig.shape[1])), mode='constant') )
        # stack and return:
        return np.stack(padded_sigs, axis=0)


    def fetch(self):
        """
        Generate and fetch a (signal, sequence, lengths) triple, where:
        * signal: a FloatTensor variable of shape (batch_size, num_levels, signal_seq_length)
        * sequence: an IntTensor variable of shape SUM{ len(seq1), len(seq2), ... }
        made up of concatenated sequences. (This is the format expected by the CTC loss operation.)
        * lengths: a 1D IntTensor variable of size (batch_size) giving the lengths of each
        sequence in the batch.
        """
        ### stop if we hit max iterations:
        self.maybe_stop()

        ### sample a list of candidate lengths:
        lengths = np.random.choice(range(self.min_length, self.max_length), size=self.batch_size)
        lengths_th = torch.IntTensor(lengths.astype(np.int32))

        ### sample a batch of random sequences:
        seqs = [np.random.randint(0, high=4, size=k, dtype=np.int32) for k in lengths]
        seq = torch.from_numpy(np.concatenate(seqs)).int()

        ### for each sequence, sample a signal sequence and stack into a batch:
        signals = [self.convert_to_signal(sq) for sq in seqs]
        signal = torch.from_numpy(self.batchify(signals)).float()

        ### update timestep
        self.tick()

        ### return on either CUDA or CPU:
        outs = (torch.autograd.Variable(signal), torch.autograd.Variable(seq), torch.autograd.Variable(lengths_th))
        if not self.on_cuda: return outs
        return (outs[0].cuda(), outs[1].cuda(), outs[2].cuda())


    # ----- Helper Functions -----
    def cuda(self):
        self.on_cuda = True

    def cpu(self):
        self.on_cuda = False

    def tick(self):
        """Update counter and maybe update epoch"""
        self.counter += 1
        if (self.counter != 0) and (self.counter % self.epoch_size == 0): self.epochs += 1

    def maybe_stop(self):
        """Return True if we are done; return False otherwise"""
        if (self.epochs == self.num_epochs) or (self.counter == self.max_iters):
            raise StopIteration
