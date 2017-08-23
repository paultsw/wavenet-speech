"""
Data pre-processing ops.
"""
import numpy as np
import argparse
from glob import glob
import os

def mu_law(x, mu=255.):
    """Apply a mu-law mapping to a floating point value."""
    return np.sign(x) * (np.log(1+mu*np.abs(x)) * np.reciprocal(np.log(1+mu)))


def mu_law_inverse(y, mu=255.):
    """Reverse a mu-law encoding."""
    return np.sign(y) * np.reciprocal(mu) * (np.power(1+mu, np.abs(y)) -1)


def clip_open(signal, window_size, threshold):
    """
    Return a clipped signal with the starting elements removed. Uses a moving window
    to detect region where the open pore signal ends and the meaningful signal begins.
    
    Args:
    * signal: the 1D NumPy array containing a continuous discrete-time signal.
    * window_size: the size of the window to move across the signal
    """
    pass


def quantize_signal(signal, num_levels, map_fn=mu_law, clip_open=False, clip_window=500, clip_threshold=200):
    """
    Take a signal and quantize it into a set of levels, i.e. discretize it.

    Args:
    * signal: a 1D NumPy array of floats.
    * num_levels: integer giving the max possible level.
    * map_fn: a function that applies a nonlinear mapping to the signal.
    * clip_open: if True, clip the start of the signal
    * clip_window: window size for clipping. (Ignored if clip_open=False.)
    * clip_threshold: threshold to call a real signal. (Ignored if clip_open=False.)
    
    Returns:
    * quantized_signal: a 1D NumPy array of integers with max value = num_levels-1, min value = 0
      (i.e. total number of levels == num_levels).
    """
    signal = clip_open(signal, clip_window, clip_threshold) if clip_open else signal

    # average/scale the signal:
    shifted_signal = (signal - np.mean(signal)) / (np.amax(signal) - np.amin(signal))

    # apply a mapping function to nonlinearly encode a signal into fixed range:
    mapping = np.vectorize(map_fn)
    mapped_signal = mapping(shifted_signal)

    # quantize into buckets:
    _bins = np.linspace(-1.0, 1.0, num=num_levels)
    quantized_signal = np.digitize(mapped_signal, _bins)

    return quantized_signal


def main(args):
    """
    Process a folder of NDArrays containing 1D signal data.
    """
    npy_glob = glob(args.input_glob)
    print("Discretizing signals. Be patient, this may take a while...")
    for npfile in npy_glob:
        discrete_signal = quantize_signal(np.load(npfile), args.levels, clip_open=args.clip)
        np.save(os.path.join(args.output_folder, npfile.strip().split("/")[-1]), discrete_signal)
    print("...All done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a folder of signals into discretized levels.")
    parser.add_argument("--levels", dest="levels", type=int, default=256, help="Number of discrete levels.")
    parser.add_argument("--clip", dest="clip", type=bool, default=False, help="Whether to clip the open pore signal.")
    parser.add_argument("--in", dest="input_glob", required=True, help="Glob of signals to process.")
    parser.add_argument("--out", dest="output_folder", required=True, help="Output folder to put quantized reads.")
    args = parser.parse_args()
    main(args)
