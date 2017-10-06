"""
A realistic model of raw signals flowing through the pore based on
current best-known models.

Typically, this would be used to e.g. pretrain a basecaller given a dataset of
(say) 10k reads in HDF5 format, by bootstrapping artificial data following the
same distributions from the HDF5 file.

For example, say you have a file called `reads.hdf5` with 10k reads. First,
dump the model distributions to NPY format:
$ python dump_distributions.py --reads reads.hdf5 --reference reference.fa

Then, use the raw signal generator below:
>> data_gen = RawSignalGenerator(...)
>> for k in range(num_training_iterations):
..   signal_data, sequence_data, lengths = data_gen.fetch()
..   model.train(signal_data, sequence_data, lengths)
"""
import torch
import numpy as np
import h5py
import random
from scipy.ndimage.filters import generic_filter

### Raw signal data generator based on best-known models:
class RawSignalGenerator(object):
    """
    An on-line random generator based on:
    * A reference genome from which we sample random lengths (converted to NPY format);
    * a numpy histogram of sample lengths per 5mer;
    * a {5mer=>Gaussian} model (e.g. from nanopolish).
    """
    def __init__(self, kmer_model, sample_counts_model, reference_hdf, read_length_model, batch_size=1):
        """
        Initialize the generator by loading all model distributions as attributes.

        Args:
        * kmer_model: path to a model file (NPZ format) containing mappings from kmers to gaussians.
        * sample_counts_model: either path to a model file (NPY) or tuple of integers; if a tuple is
        provided, use those as the bounds of a uniform distribution for the number of raw samples per 5mer.
        * reference_hdf: path to an HDF5 fiel containing a reference genome, from which random reads
        are drawn; positions on the reference genome are drawn uniformly at random.
        * read_length_model: either path to a model file (NPY) or tuple of integers; if a tuple is
        provided, use those as the bounds of a uniform distribution from which the read length is drawn.
        * batch_size: number of data/target sequences per batch.
        """
        # save input params:
        self.batch_size = batch_size
        self.kmer_model = kmer_model
        self.sample_counts_model = sample_counts_model
        self.reference_hdf = reference_hdf
        self.read_length_model = read_length_model

        # load gaussian model:
        self.num_kmers = 4**5 # total number of 5-mers
        kmer_model_npz = np.load(kmer_model)
        self.kmer_means = kmer_model_npz['means']
        self.kmer_stdvs = kmer_model_npz['stdvs']

        # load reference genome as file handle to HDF5:
        self.reference = h5py.File(reference_hdf, 'r')
        self.contigs = list(self.reference.keys())

        # load read lengths model and normalize:
        if isinstance(read_length_model, tuple):
            self.read_lengths = np.zeros(read_length_model[1])
            for k in range(read_length_model[0], read_length_model[1]):
                self.read_lengths[k] = 1.
            self.read_lengths = self.read_lengths / np.sum(self.read_lengths)
        else:
            self.read_lengths = np.load(read_length_model)
            self.read_lengths = self.read_lengths / np.sum(self.read_lengths)

        # load sample lengths model and normalize:
        if isinstance(sample_counts_model, tuple):
            self.sample_lengths = np.zeros(sample_counts_model[1])
            for k in range(sample_counts_model[0], sample_counts_model[1]):
                self.sample_lengths[k] = 1.
            self.sample_lengths = self.sample_lengths / np.sum(self.sample_lengths)
        else:
            self.sample_lengths = np.load(sample_counts_model)
            self.sample_lengths = self.sample_lengths / np.sum(self.sample_lengths)

        # define lambda function to convert kmers to integer indices:
        self.nts_to_kmer = lambda nts: np.sum((nts-np.ones(nts.shape)) * np.array([256, 64, 16, 4, 1]))


    def close(self):
        """Close file handle."""
        self.reference.close()


    def gaussian_model_fn(self, sequence):
        """
        Converts a sequence of nucleotides into a sequence of pico-amps by gaussian lookup of
        the underlying 5mers using the kmer model.
        """
        # extract list of kmers from moving window across sequence as integer in [0,1023]:
        kmer_seq = generic_filter(sequence, self.nts_to_kmer, size=(5,), mode='constant')
        kmer_seq = kmer_seq[4:-4].astype(int) # (remove, since first/last 4 values are padded)

        # upsample the kmer sequence according to the sample model:
        kmer_seq = random_upsample(kmer_seq, self.sample_lengths, axis=0)

        # look up corresponding values in kmer_means, kmer_stdvs:
        kmer_means = np.array([self.kmer_means[k] for k in kmer_seq])
        kmer_stdvs = np.array([self.kmer_stdvs[k] for k in kmer_seq])

        # generate gaussians:
        gaussian_signals = np.random.normal(loc=kmer_means, scale=kmer_stdvs)
        
        # return:
        return gaussian_signals


    def fetch(self):
        """
        Generate and fetch a (signal, sequence, lengths) triple, where:
        * signal: a FloatTensor variable of shape (batch_size, num_levels, signal_seq_length)
        * sequence: an IntTensor variable of shape SUM{ len(seq1), len(seq2), ... }
        made up of concatenated sequences. (This is the format expected by the CTC loss operation.)
        * lengths: a 1D IntTensor variable of size (batch_size) giving the lengths of each
        sequence in the batch.
        """
        ### sample a list of candidate lengths:
        lengths = sample_from_pmf(self.read_lengths, size=self.batch_size)
        lengths_th = torch.IntTensor(lengths.astype(np.int32))

        ### sample a batch of random sequences:
        seqs = [fetch_from_reference(self.reference, self.contigs, k) for k in lengths]
        seq = torch.from_numpy(np.concatenate(seqs)).int()

        ### for each sequence, sample a signal sequence and stack into a batch:
        signals = [self.gaussian_model_fn(sq) for sq in seqs]
        signal = torch.from_numpy(batchify(signals)).float()

        ### return as torch.autograd.Variable on CPU:
        outs = (torch.autograd.Variable(signal), torch.autograd.Variable(seq), torch.autograd.Variable(lengths_th))
        return outs


##### Helper functions:
def sample_from_pmf(pmf_array, size=1):
    """Independently draw `size` examples from a probability mass given by `pmf_array`; return as np array."""
    return np.random.choice(np.arange(pmf_array.shape[0]), p=pmf_array, size=size)

def random_upsample(label_seq, repeat_model, axis=0):
    """Randomly repeat each component in a label sequence according to a distribution of repeats."""
    num_repeats = sample_from_pmf(repeat_model, size=label_seq.shape)
    return np.repeat(label_seq, num_repeats, axis=axis)

def fetch_from_reference(ref, contigs, L):
    """Select a random subinterval of length L from a reference genome `ref`."""
    # choose random contig:
    ctg = ref[random.choice(contigs)]['contig']
    # choose random position:
    pos = np.random.randint(ctg.shape[0]-L)
    return ctg[pos:(pos+L)]

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
