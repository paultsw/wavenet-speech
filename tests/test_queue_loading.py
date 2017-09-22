"""
Test multiprocessing queue loader.
"""
import torch
import numpy as np
from utils.loaders import QueueLoader
import traceback
import sys

### construct queue loader on E.Coli dataset:
dataset_path = "data/ecoli/reads.01.reference.hdf5"
loader = QueueLoader(dataset_path, num_workers=5, max_iters=100, epoch_size=10)

### test settings:
num_prints = 300

### loop over the queue and print length statistics
try:
    for k in range(num_prints):
        vals = loader.dequeue()
        print(k, "Signal shape: ", vals[0].size(), "Total sequence lengths: ", vals[1].size())
    print("Done")
    loader.close()
### handle errors and close queue:
except StopIteration:
    print("All iterations finished; StopIteration called.")
    print("Num iterations: {}".format(loader.global_counter))
    print("Num epochs: {}".format(loader.num_epochs))
except:
    traceback.print_exc()
finally:
    loader.close()
    sys.exit()
