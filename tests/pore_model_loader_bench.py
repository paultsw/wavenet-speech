"""
Speed-test benchmark for infinite-data pore-model generator.
"""
import torch
import numpy as np
import timeit
from utils.pore_model import PoreModelLoader

pore_model_generator = PoreModelLoader(100, 1, 100, batch_size=10)

def fetch_datapoints():
    """
    Fetch a single triple of torch variables from the pore loader.
    """
    _ = pore_model_generator.fetch()
    return

if __name__ == '__main__':
    print(timeit.timeit("fetch_datapoints()", number=1, setup="from __main__ import fetch_datapoints"))
