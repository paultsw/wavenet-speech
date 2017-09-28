"""
Pore Generator Loader class.
"""
import torch
import numpy as np
from random import randint, shuffle
from scipy.ndimage.filters import generic_filter

class GaussianModelLoader(object):
    """
    An on-line random generator based on Nanopolish's gaussian 5mer model.
    """
    def __init__(self, max_iters, num_epochs, epoch_size, kmer_model_path,
                 batch_size=1, num_levels=256, upsampling=3, lengths=(20,30)):
        """
        Initialize the gaussian 5mer model loader. Reads a pickled 5mer model
        based off of nanopolish's r9.4 5mer template model.
        """
        # save input params:
        self.max_iters = max_iters
        self.num_epochs = num_epochs
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.num_levels = num_levels
        self.min_length = lengths[0]
        self.max_length = lengths[1]
        self.upsampling = upsampling
        self.path_to_model = kmer_model_path

        # internal record-keeping parameters:
        self.counter = 0
        self.epochs = 0
        self.on_cuda = False

        # define quantization law and levels:
        _mu = float(num_levels)
        _mu_law = lambda x: (np.sign(x) * (np.log(1+_mu*np.abs(x)) * np.reciprocal(np.log(1+_mu))))
        self.quant_law = np.vectorize(_mu_law)
        self.quant_levels = np.linspace(-1.0, 1.0, num=num_levels)

        # load gaussian model:
        self.num_kmers = 4**5 # total number of 5-mers
        kmer_model_npz = np.load(kmer_model_path)
        self.kmer_means = kmer_model_npz['means']
        self.kmer_stdvs = kmer_model_npz['stdvs']

        # define lambda function to convert kmers to integer indices:
        self.nts_to_kmer = lambda nts: np.sum((nts-np.ones(nts.shape)) * np.array([256, 64, 16, 4, 1]))


    def gaussian_model_fn(self, sequence):
        """
        Converts a sequence of nucleotides into a sequence of pico-amps by gaussian lookup.
        """
        # extract list of kmers from moving window across sequence as integer in [0,1023]:
        kmer_seq = generic_filter(sequence, self.nts_to_kmer, size=(5,), mode='constant')
        kmer_seq = kmer_seq[4:-4].astype(int) # (remove, since first/last 4 values are padded)

        # upsample the kmer sequence:
        if (self.upsampling > 1): kmer_seq = kmer_seq.repeat(self.upsampling, axis=0)

        # look up corresponding values in kmer_means, kmer_stdvs:
        kmer_means = np.array([self.kmer_means[k] for k in kmer_seq])
        kmer_stdvs = np.array([self.kmer_stdvs[k] for k in kmer_seq])

        # generate gaussians:
        gaussian_signals = np.random.normal(loc=kmer_means, scale=kmer_stdvs)

        # upsample and return:
        return gaussian_signals


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
        picoamp_signal = self.gaussian_model_fn(seq)
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




### Raw signal level gaussian model
class RawGaussianModelLoader(object):
    """
    An on-line random generator based on Nanopolish's gaussian 5mer model.
    """
    def __init__(self, max_iters, num_epochs, epoch_size, kmer_model_path,
                 batch_size=1, upsampling=3, lengths=(20,30)):
        """
        Initialize the gaussian 5mer model loader. Reads a pickled 5mer model
        based off of nanopolish's r9.4 5mer template model.
        """
        # save input params:
        self.max_iters = max_iters
        self.num_epochs = num_epochs
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.min_length = lengths[0]
        self.max_length = lengths[1]
        self.upsampling = upsampling
        self.path_to_model = kmer_model_path

        # internal record-keeping parameters:
        self.counter = 0
        self.epochs = 0
        self.on_cuda = False

        # load gaussian model:
        self.num_kmers = 4**5 # total number of 5-mers
        kmer_model_npz = np.load(kmer_model_path)
        self.kmer_means = kmer_model_npz['means']
        self.kmer_stdvs = kmer_model_npz['stdvs']

        # define lambda function to convert kmers to integer indices:
        self.nts_to_kmer = lambda nts: np.sum((nts-np.ones(nts.shape)) * np.array([256, 64, 16, 4, 1]))


    def gaussian_model_fn(self, sequence):
        """
        Converts a sequence of nucleotides into a sequence of pico-amps by gaussian lookup.
        """
        # extract list of kmers from moving window across sequence as integer in [0,1023]:
        kmer_seq = generic_filter(sequence, self.nts_to_kmer, size=(5,), mode='constant')
        kmer_seq = kmer_seq[4:-4].astype(int) # (remove, since first/last 4 values are padded)

        # look up corresponding values in kmer_means, kmer_stdvs:
        kmer_means = np.array([self.kmer_means[k] for k in kmer_seq])
        kmer_stdvs = np.array([self.kmer_stdvs[k] for k in kmer_seq])

        # generate gaussians:
        gaussian_signals = np.random.normal(loc=kmer_means, scale=kmer_stdvs)

        # upsample and return:
        signal = gaussian_signals.repeat(self.upsampling, axis=0) if (self.upsampling > 1) else gaussian_signals
        return signal


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
        pad_length = max([sig.shape[0] for sig in signals_list])
        # pad all ndarrs and stack together in 0-axis:
        padded_sigs = []
        for sig in signals_list:
            padded_sigs.append( np.pad(sig, (0,pad_length-sig.shape[0]), mode='constant') )
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
        seq = torch.from_numpy(np.concatenate(seqs)).int()

        ### for each sequence, sample a signal sequence and stack into a batch:
        signals = [self.gaussian_model_fn(sq) for sq in seqs]
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
