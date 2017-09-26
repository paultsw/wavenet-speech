"""
Model evaluation script for wavenet-speech.
"""
import torch
import torch.nn as nn

from modules.wavenet import WaveNet
from modules.classifier import WaveNetClassifier
from modules.sequence_decoders import argmax_decode, BeamSearchDecoder # [TODO: update]

import os
import argparse
from utils.config_tools import json_to_config, config_to_json
from utils.loaders import Loader # [TODO: update]


def main(args):
    """
    Main function; performs the following:

    1) reconstructs a wavenet classifier model from configuration;
    2) restores weights (note that the weights *must* match the architecture specified in the configuration);
    3) loads a test dataset;
    4) benchmark the model's performance on a collection of random reads from the dataset.
    """
    # [TODO]
    return None


if __name__ == '__main__':
    # load args:
    parser = argparse.ArgumentParser(description="Evaluate a trained WaveNet classifier model.")
    parser.add_argument("--config_path", required=True, help="Path to configuration file defining the model architecture.")
    parser.add_argument("--wavenet_model_path", help="Path to model weights for WaveNet base.")
    parser.add_argument("--classifier_model_path", help="Path to model weights for WaveNet Classifier layers.")
    parser.add_argument("--print_examples", default=False, help="If set, print the outputs alongside the actual targets.")
    parser.add_argument("--dataset", required=True, help="Path to test dataset (in HDF5 format).")
    parser.add_argument("--num_examples", required=True, default=100)
    parser.parse_args()
