"""
Read an HDF5 file and a reference genome FASTA file and dump empirical distributions
and genome data to NPY/HDF5 files.
"""
import numpy as np
import pysam
from tqdm import tqdm
import argparse
import os
import h5py
from collections import defaultdict
import traceback


_DEFAULT_NTDICT_ = {
    'A': 1, 'G': 2, 'C': 3, 'T': 4,
    'a': 1, 'g': 2, 'c': 3, 't': 4,
}
def string_to_array(nts, lookup_dict=_DEFAULT_NTDICT_):
    """Convert a string of ACGT/acgt to integer labels."""
    labels = [lookup_dict[ch] for ch in nts if (ch in lookup_dict.keys())]
    return np.array(labels, dtype=np.int32)


def extract_and_dump(reads_path, reference_path, outdir):
    """
    Main function: dump reference genome as a (looooong) numpy array
    and get statistical distributions from reads.
    """
    ### Read reference genome and dump to HDF5; ignore all 'N's.
    print("Parsing reference genome to HDF5 format...")
    ref_fa = None
    ref_hdf = None
    try:
        # open fasta file:
        ref_fa = pysam.FastaFile(reference_path)
        contigs = ref_fa.references
        # open HDF5 file:
        ref_hdf = h5py.File(os.path.join(outdir,"reference.hdf5"), 'x')
        for ctg in contigs:
            contig_string = ref_fa.fetch(reference=ctg, start=None, end=None)
            contig_arr = string_to_array(contig_string)
            contig_grp = ref_hdf.create_group(ctg.strip())
            contig_dset = contig_grp.create_dataset("contig", data=contig_arr)
            contig_dset.attrs['size'] = contig_arr.shape[0]
            print("Finished contig: {}".format(ctg))
        print("... Done.")
    except:
        traceback.print_exc()
    finally:
        if not (ref_fa is None): ref_fa.close()
        if not (ref_hdf is None): ref_hdf.close()


    ### Get all distributional statistics from HDF5 file:
    print("Getting read lengths and per-kmer sample counts...")
    hf = h5py.File(reads_path, 'r')
    try:
        # accumulate count and read length distributions:
        sample_counts = defaultdict(int)
        read_lengths = defaultdict(int)
        max_count = -1
        max_length = -1
        for read in tqdm(hf.keys()):
            raw_samples = hf[read]['raw']['samples'][:]
            for arr in raw_samples: sample_counts[arr.shape[0]] += 1
            max_count = max(max_count, arr.shape[0])
            read_size = hf[read]['reference'][:].shape[0]
            read_lengths[read_size] += 1
            max_length = max(max_length, read_size)
        # dump to numpy array:
        counts_list = [sample_counts[k] for k in range(1,max_count+1)]
        counts_arr = np.array(counts_list, dtype=np.int32)
        lengths_list = [read_lengths[k] for k in range(1,max_length+1)]
        lengths_arr = np.array(lengths_list, dtype=np.int32)
        np.save(os.path.join(outdir,'sample_counts.npy'), counts_arr)
        np.save(os.path.join(outdir,'read_lengths.npy'), lengths_arr)
        print("...Done.")
    except:
        traceback.print_exc()
    finally:
        print("Closing HDF5 file...")
        hf.close()
        print("...Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dump reference and reads to NPY format.")
    parser.add_argument("--reads", dest="reads", required=True, help="Path to reads data in HDF5 format.")
    parser.add_argument("--reference", dest="reference", required=True, help="Path to reference genome in FASTA format.")
    parser.add_argument("--outdir", dest="outdir", default="./", help="Folder to place the NPY files in.")
    args = parser.parse_args()
    assert os.path.exists(args.reads)
    assert os.path.exists(args.reference)
    assert os.path.exists(args.outdir)
    extract_and_dump(args.reads, args.reference, args.outdir)
