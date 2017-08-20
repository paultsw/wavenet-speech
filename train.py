"""
Train a new WaveNet speech recognition module.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

# custom modules:
from modules.wavenet import WaveNet
from modules.classifier import WaveNetClassifier
from utils.loader import Loader

# argument parsers:
import argparse
import os

def wavenet_train_step(wave, classifier, nll_fn, ctc_fn):
    """
    Take one training step of WaveNet.
    """
    pass

def train():
    """
    Main training loop.
    """
    # construct wavenet:
    wavenet_core = WaveNet()
    wavenet_class = WaveNetClassifier()
    
    # construct loss functions:
    nll_loss_fn = nn.NLLLoss()
    ctc_loss_fn = CTCLoss()

    # construct optim:
    opt = optim.RMSprop() # FIX - CHOOSE BETTER OPTIM

    # construct data loader:
    pass
    
    # run training loop:
    pass


if __name__ == '__main__':
    # read CLI args:
    parser = argparse.ArgumentParser()
