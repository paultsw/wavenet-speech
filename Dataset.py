from utils.gaussian_kmer_model import GaussianModelLoader, RawGaussianModelLoader
from utils.pore_model import PoreModelLoader
from utils.loaders import QueueLoader
import torch
import numpy as np


# default mapping from nucleotides to underlying pico-amp values:
_CURRS_ = { 1: 51., 2: 22., 3: 103., 4: 115. }
_KMER_MODEL_PATH_ = "./utils/r9.4_450bps.5mer.template.npz"

class Dataset(object):
    """
    Container class for queue and artificial datasets.
    """
    def __init__(self, datatype, dataset=None):
        """
        Construct some sort of dataset.

        Allowed datatypes: '(raw-)pore' (pore model), '(raw-)gauss' (nanopolish model), '(raw-)hdf5' (hdf5 file + queue).
        """
        assert (datatype in ['pore', 'gauss', 'hdf5', 'raw-pore', 'raw-gauss', 'raw-hdf5'])
        self.datatype = datatype
        self.dataset = dataset

        max_iters = 1000
        num_epochs = 1
        epoch_size = 1000
        bsz = 8
        nlevels = 256
        min_len = 90
        max_len = 100
        upsample = 4
        noise = 2.0
        pw = 4
        nworkers = 1
        qsize = 50

        # construct dataset dependent on which one we want:
        if datatype == 'pore':
            self.data = PoreModelLoader(max_iters, num_epochs, batch_size=bsz, num_levels=nlevels, lengths=(min_len,max_len),
                                        pore_width=pw, sample_rate=upsample, currents_dict=_CURRS_, sample_noise=noise,
                                        interleave_blanks=False, raw_signal=False)
        if datatype == 'raw-pore':
            self.data = PoreModelLoader(max_iters, num_epochs, batch_size=bsz, num_levels=nlevels, lengths=(min_len,max_len),
                                        pore_width=pw, sample_rate=upsample, currents_dict=_CURRS_, sample_noise=noise,
                                        interleave_blanks=False, raw_signal=True)
        if datatype == 'gauss':
            self.data = GaussianModelLoader(max_iters, num_epochs, epoch_size, _KMER_MODEL_PATH_, batch_size=bsz,
                                            num_levels=nlevels, upsampling=upsample, lengths=(min_len,max_len))
        if datatype == 'raw-gauss':
            self.data = RawGaussianModelLoader(max_iters, num_epochs, epoch_size, _KMER_MODEL_PATH_, batch_size=bsz,
                                               upsampling=upsample, lengths=(min_len,max_len))
        if datatype == 'hdf5':
            self.data = QueueLoader(dataset, num_signal_levels=nlevels, num_workers=nworkers, queue_size=qsize,
                                    batch_size=bsz, sample_lengths=(min_len,max_len), max_iters=max_iters, epoch_size=epoch_size)
        if datatype == 'raw-hdf5':
            raise NotImplementedError("Raw HDF5 signal data currently unsupported!")


    def fetch(self, train_or_valid='train'):
        """Returns a tuple of data of the form (signal, sequence, sequence_lengths)."""
        if (self.datatype in ['pore', 'gauss', 'raw-pore', 'raw-gauss']):
            signal, sequence , seqlengths = self.data.fetch()
        if (self.datatype in ['hdf5', 'raw-hdf5']):
            signal, sequence, _, seqlengths = self.data.dequeue(from_queue=train_or_valid)
        return (signal, sequence, seqlengths)


    def close(self):
        """Close the internal dataset; only does anything if we've opened an HDF5 handle."""
        if self.datatype in ['hdf5', 'raw-hdf5']:
            self.data.close()
