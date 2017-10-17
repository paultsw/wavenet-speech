"""
Read an eventaligned TSV file with duration data per 5mer and create a gamma-distribution model
based on observed durations.
"""
import numpy as np
import csv
import os
from collections import deque, defaultdict

from scipy.special import digamma, polygamma
import scipy.stats as stats

class DurationModel(object):
    """Class to hold our duration model data."""
    def __init__(self):
        """Initialize internal values."""
        # construct 5mer model distribution with prior parameters
        self.model_parameters = {}
        for k in range(4**5): # 4^5 == #{5mers}
            self.model_parameters[k] = {'shape': 2.461964, 'rate': 587.2858}
        # nucleotide lookup dictionary:
        self.nt2idx_dict = {'A': 0, 'G': 1, 'C': 2, 'T': 3,
                            'a': 0, 'g': 1, 'c': 2, 't': 3}
        self.idx2nt_dict = {0: 'A', 1: 'G', 2: 'C', 3: 'T'}
        # holders for raw sample data:
        self.samples = defaultdict(list)

    def str2idx(self, kmer_str):
        """Return 5mer index from 0-1023."""
        return sum([self.nt2idx_dict[kmer_str[k]]*(4**k) for k in range(0,5)])
    
    def idx2str(self, kmer_idx):
        """Return the identity of the 5mer with a given index."""
        k4 = kmer_idx // 4**4
        k3 = (kmer_idx - k4*(4**4)) // 4**3
        k2 = (kmer_idx - k3*(4**3) - k4*(4**4)) // 4**2
        k1 = (kmer_idx - k2*(4**2) - k3*(4**3) - k4*(4**4)) // 4
        k0 = (kmer_idx - k1*4 - k2*(4**2) - k3*(4**3) - k4*(4**4))
        return ("".join([self.idx2nt_dict[k0],
                         self.idx2nt_dict[k1],
                         self.idx2nt_dict[k2],
                         self.idx2nt_dict[k3],
                         self.idx2nt_dict[k4]]))

    def dump_model(self, save_path):
        """
        Dump the internal model to NPY file at some specified path. The NPY-formatted model is of shape
        `[1024, 2]` where the first column is the shape and the second column is the rate; `dtype == np.float32`.
        """
        shape_vec_npy = np.array([self.model_parameters[k]['shape'] for k in range(1024)])
        rate_vec_npy = np.array([self.model_parameters[k]['rate'] for k in range(1024)])
        model_parameters_npy = np.stack((shape_vec_npy, rate_vec_npy), axis=1)
        np.save(save_path, model_parameters_npy)

    def update_kmer(self, kmer_idx, samples):
        """Use MLE to update the model parameters for some kmer. `samples` is a numpy array of observed event durations."""
        # look up values:
        shape, rate = self.model_parameters[kmer_idx]['shape'], self.model_parameters[kmer_idx]['rate']
        # perform MLE by calling helper subroutine:
        alpha, loc, beta = stats.gamma.fit(samples)
        self.model_parameters[kmer_idx]['shape'] = alpha
        self.model_parameters[kmer_idx]['rate'] = beta

    def update_all_kmer_models(self):
        """For each kmer, perform maximum likelihood training on the parameters."""
        for k in range(1024):
            self.update_kmer(k, self.samples[k])


def maybe_append(rows, duration_mdl):
    """
    Appends the middle sample of `rows` (a deque) to `duration_mdl.samples` if all of the following criteria are met:
    1. previous and next events are *not* in the same reference position;
    2. the reference kmer is *not* 'NNNNN';
    3. previous and next event indices are *not* in the same position;
    4. make sure the dequeue is full.
    """
    # don't append anything if the rows deque is not full:
    if len(rows) < 3: pass

    # get all relevant data:
    positions = int(rows[0]['position']), int(rows[1]['position']), int(rows[2]['position'])
    reference_kmers = rows[0]['reference_kmer'], rows[1]['reference_kmer'], rows[2]['reference_kmer']
    evt_idxs = int(rows[0]['event_index']), int(rows[1]['event_index']), int(rows[2]['event_index'])
    evt_duras = float(rows[0]['event_length']), float(rows[1]['event_length']), float(rows[2]['event_length'])

    # skip if middle ref has an 'N' in it:
    if ('N' in reference_kmers[1]) or ('n' in reference_kmers[1]): pass

    # skip if no change in reference position:
    if (positions[0] == positions[1]) or (positions[1] == positions[2]): pass

    # skip if no change in event indices:
    if (evt_idxs[0] == evt_idxs[1]) or (evt_idxs[1] == evt_idxs[2]): pass
    
    # if you've made it this far, it's finally safe to append the duration:
    duration_mdl.samples[duration_mdl.str2idx[reference_kmers[1]]].append(evt_duras[1])


def main(eventalign_tsv_file, npy_dump_path):
    """
    Main loop: go through eventaligned TSV file and accumulate admissible rows; then, use MLE updates to estimate
    optimal parameters.
    """
    # construct duration model:
    durations = DurationModel()
    
    # loop over (possibly LARGE! ~ 200GB) eventalign file and collect samples:
    print("Appending samples...")
    assert os.path.exists(eventalign_tsv_file) # sanity check
    eaf_headers = ['contig', 'position', 'reference_kmer', 'read_index',
                   'strand', 'event_index', 'event_level_mean', 'event_stdv', 'event_length',
                   'event_start_time', 'model_kmer', 'model_mean', 'model_stdv', 'standardized_level']
    with open(eventalign_tsv_file, 'r') as eaf:
        rdr = csv.DictReader(eaf, delimiter='\t', fieldnames=eaf_headers, quoting=csv.QUOTE_NONE)
        next(rdr,None) # skip header
        # row buffer of fixed width 3:
        row_buffer = deque(maxlen=3)
        for row in rdr:
            row_buffer.append(row)
            maybe_append(row_buffer, durations)
    print("...Done.")

    # update all parameters in duration model and dump to numpy:
    print("Maximum likelihood training on all kmer parameters...")
    durations.update_all_kmer_models()
    durations.dump_model(npy_dump_path)
    print("...Done. Dumped gamma models to: {}".format(npy_dump_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sharpen a prior on a gamma-distributed duration model for 5mers.")
    parser.add_argument("eventalign_file")
    parser.add_argument("npy_dump_path")
    args = parser.parse_args()
    main(args.eventalign_file, args.npy_dump_path)
