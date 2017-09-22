"""
Pore Generator Loader class.
"""
import torch
import numpy as np
from random import randint, shuffle
from scipy.ndimage.filters import generic_filter
from scipy.signal import triang

# default mapping from nucleotides to underlying pico-amp values:
_CURRS_ = { 1: 51., 2: 22., 3: 103., 4: 115. }
class PoreModelLoader(object):
    """
    An on-line random generator for artificial pore data.
    """
    def __init__(self, max_iters, num_epochs, epoch_size, batch_size=1, num_levels=256,
                 lengths=(20,30), pore_width=4, sample_rate=3, currents_dict=_CURRS_,
                 sample_noise=3.0, interleave_blanks=False):
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
        * interleave_blanks: if True, return blank symbols (0) between each nucleotide ({1-4}) in target seqs.
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
        self.interleave_blanks = interleave_blanks

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
        seqs = [np.random.randint(1, high=5, size=k, dtype=np.int32) for k in lengths]

        ### for each sequence, sample a signal sequence and stack into a batch:
        signals = [self.convert_to_signal(sq) for sq in seqs]
        signal = torch.from_numpy(self.batchify(signals)).float()

        ### optionally interleave zeros into the output:
        if self.interleave_blanks:
            seqs = [self.interleave_zeros(seq) for seq in seqs]
            lengths_th.mul_(2)
        seq = torch.from_numpy(np.concatenate(seqs)).int()

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

    @staticmethod
    def interleave_zeros(ndarr):
        """Puts a 0 after each value in the np.int32 ndarray."""
        zeros_vec = np.zeros(ndarr.shape[0], dtype=np.int32)
        return np.ravel(np.column_stack((ndarr,zeros_vec)))
