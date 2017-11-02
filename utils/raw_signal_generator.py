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
    * A reference genome (in HDF5 format) from which we sample reads of random lengths;
    * a {5mer=>Gamma(a,b)} model of num. samples per 5mer;
    * a {5mer=>Gaussian(mu,sigma)} model of picoamp mean/stdv per 5mer (e.g. from nanopolish's r9 models).
    
    You can generate each of the above by running the appropriate scripts in this directory.
    """
    def __init__(self, kmer_model, reference_hdf, read_length_model, sample_rate=800., batch_size=1,
                 dura_shape=None, dura_rate=None, pad_label=0):
        """
        Initialize the generator by loading all model distributions as attributes.

        Args:
        * kmer_model: path to a model file (NPZ format) containing mappings from kmers to gaussians.
        * reference_hdf: path to an HDF5 file containing a reference genome, from which random reads
        are drawn; positions on the reference genome are drawn uniformly at random.
        * read_length_model: either path to a model file (NPY) or tuple of integers; if a tuple is
        provided, use those as the bounds of a uniform distribution from which the read length is drawn.
        * sample_rate: number of samples per second to simulate; should be >> 450 for realism.
        * batch_size: number of data/target sequences per batch.
        * dura_shape, dura_rate: if these are provided and are floats, these will override the default
        parameters of the duration model.
        * pad_label: the integer label of the padding value for the underlying nucleotide sequence.
        (N.B.: padding labels other than '0' are currently unsupported and will raise an error. Support
        is planned for further down the roadmap.)
        """
        # save input params:
        self.batch_size = batch_size
        self.kmer_model = kmer_model
        self.reference_hdf = reference_hdf
        self.read_length_model = read_length_model
        self.sample_rate = sample_rate
        self.dura_shape_arg = dura_shape
        self.dura_rate_arg = dura_rate
        self.pad_label = pad_label
        if pad_label != 0: raise ValueError("ERR: padding values other than 0 are currently unsupported.")

        # load gaussian model:
        self.num_kmers = 4**5 # total number of 5-mers
        kmer_model_npz = np.load(kmer_model)
        self.kmer_means = kmer_model_npz['means']
        self.kmer_stdvs = kmer_model_npz['stdvs']

        # load reference genome as file handle to HDF5:
        self.reference = h5py.File(reference_hdf, 'r')
        self.contigs = list(self.reference.keys())

        # hard-coded shape/rate parameters for gamma-distributed duration modelling:
        self.sample_rate = sample_rate
        self.duration_shape = 2.461964 if dura_shape is None else dura_shape
        self.duration_rate = 587.2858 if dura_rate is None else dura_rate

        # load read lengths model and normalize:
        if isinstance(read_length_model, tuple):
            self.read_lengths = np.zeros(read_length_model[1])
            for k in range(read_length_model[0], read_length_model[1]):
                self.read_lengths[k] = 1.
            self.read_lengths = self.read_lengths / np.sum(self.read_lengths)
        else:
            self.read_lengths = np.load(read_length_model)
            self.read_lengths = self.read_lengths / np.sum(self.read_lengths)


    def nts_to_kmer(self, nts):
        """Define lambda function to convert kmers to integer indices."""
        return np.sum((nts-np.ones(nts.shape)) * np.array([256, 64, 16, 4, 1]))


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
        kmer_seq = kmer_seq[2:-2].astype(np.int64) # remove kmers that rely on paddings

        # upsample the kmer sequence according to the sample model:
        kmer_seq = random_upsample(kmer_seq, self.duration_shape, self.duration_rate, self.sample_rate, axis=0)

        # look up corresponding values in kmer_means, kmer_stdvs:
        kmer_means = np.array([self.kmer_means[k] for k in kmer_seq])
        kmer_stdvs = np.array([self.kmer_stdvs[k] for k in kmer_seq])

        # generate gaussians:
        gaussian_signals = np.random.normal(loc=kmer_means, scale=kmer_stdvs)
        
        # return:
        return gaussian_signals


    def fetch(self):
        """
        Generate and fetch a (signal, sequence, sig_lengths, seq_lengths) tuple, where:
        * signal: a FloatTensor variable of shape (batch_size, signal_seq_length) representing the raw
        floating point picoamp values sampled from the gaussian 5mer model.
        * sequence: an IntTensor variable of shape SUM{ len(seq1), len(seq2), ... }
        made up of concatenated sequences. (This is the format expected by the CTC loss operation.)
        * sig_lengths: a 1D IntTensor variable of size (batch_size) giving the lengths of each
        signal in the batch.
        * seq_lengths: a 1D IntTensor variable of size (batch_size) giving the lengths of each
        sequence in the batch.
        """
        ### sample a list of candidate lengths:
        seq_lengths = sample_from_pmf(self.read_lengths, size=self.batch_size)
        seq_lengths_th = torch.IntTensor(seq_lengths.astype(np.int32))

        ### sample a batch of random sequences:
        seqs = [fetch_from_reference(self.reference, self.contigs, k) for k in seq_lengths]
        seq = torch.from_numpy(batchify(seqs)).long()

        ### for each sequence, sample a signal sequence and stack into a batch:
        signals = [self.gaussian_model_fn(sq) for sq in seqs]
        signal = torch.from_numpy(batchify(signals)).float()
        sig_lengths = [sgl.shape[-1] for sgl in signals]
        sig_lengths_th = torch.IntTensor(np.array(sig_lengths, dtype=np.int32))

        ### return as torch.autograd.Variable on CPU:
        outs = (torch.autograd.Variable(signal),
                torch.autograd.Variable(seq, requires_grad=False),
                torch.autograd.Variable(sig_lengths_th),
                torch.autograd.Variable(seq_lengths_th, requires_grad=False))
        return outs


##### Helper functions:
def sample_from_pmf(pmf_array, size=1):
    """Independently draw `size` examples from a probability mass given by `pmf_array`; return as np array."""
    return np.random.choice(np.arange(pmf_array.shape[0]), p=pmf_array, size=size)

def fetch_from_reference(ref, contigs, L):
    """Select a random subinterval of length L from a reference genome `ref`."""
    # choose random contig:
    ctg = ref[random.choice(contigs)]['contig']
    # choose random position:
    pos = np.random.randint(ctg.shape[0]-L)
    return ctg[pos:(pos+L)]

def batchify(seqs_list):
    """
    Pad a list to uniform length with vectors consisting solely of zeros.
    
    Args:
    * seqs_list: a list of np.float32 or np.int64 arrays, each of shape (seq_length,).
    Returns:
    * seqs_batch: np.float32 of shape ( len(seqs_list), max[seqs_length] ).
    """
    # get max length:
    pad_length = max([seq.shape[0] for seq in seqs_list])
    # pad all ndarrs and stack together in 0-axis:
    padded_seqs = []
    for seq in seqs_list:
        padded_seqs.append( np.pad(seq, (0, pad_length-seq.shape[0]), mode='constant') )
    # stack and return:
    return np.stack(padded_seqs, axis=0)

def random_upsample(label_seq, gamma_shape, gamma_rate, srate, axis=0):
    """
    Randomly repeat each component in a label sequence according to a gamma distribution of sample durations.

    Args:
    * label seq: a sequence of integer labels, to be repeated.
    * gamma_shape: `shape` parameter of the gamma distribution
    * gamma_rate: the rate parameter of the duration.
    * srate: the sample rate.
    
    Returns: `label_seq` with each label repeated a random number of times.
    """
    num_repeats = (np.random.gamma(gamma_shape, np.reciprocal(gamma_rate), size=label_seq.shape) * srate).astype(np.int32)
    num_repeats = num_repeats + (num_repeats == 0).astype(np.int32) # enforce num_repeats >= 1
    return np.repeat(label_seq, num_repeats, axis=axis)
