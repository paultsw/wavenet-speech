"""
Data-loading class.
"""
import torch
import torch.utils.data as data
import numpy as np

class Loader(object):
    """
    Signal and base-sequence loader.

    [TODO: finish this; primary challenge is the difference in temporal dimensionality of the sequences.
     Consider loading batches of approximately the same length?]
    """
    def __init__(self):
        pass

    def fetch(self):
        pass


class OverfitLoader(object):
    """
    Loads a single (signal, base-sequence) pair over and over again.
    """
    def __init__(self, signal_path, sequence_path, batch_size=16, dataset_size=16, num_levels=256):
        """Construct a tensor dataset."""
        signal_tensor = torch.from_numpy(np.load(signal_path)).unsqueeze(1)
        one_hot_signal_tensor = torch.zeros(signal_tensor.size(0), num_levels)
        one_hot_signal_tensor.scatter_(1, signal_tensor, 1.)

        sequence_tensor = torch.from_numpy(np.load(sequence_path))
        
        # stack the signal tensor multiple times:
        stacked_signal_tensor = torch.stack([one_hot_signal_tensor] * dataset_size, dim=0).unsqueeze(2)

        # stack the sequence tensor multiple times:
        stacked_sequence_tensor = torch.stack([sequence_tensor] * dataset_size, dim=0)

        # torch dataset:
        self.dataset = data.TensorDataset(stacked_signal_tensor, stacked_sequence_tensor)
        
        # torch data loader:
        self.loader = data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
