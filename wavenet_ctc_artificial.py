"""
Train WaveNet-CTC to overfit on artificial signal-to-sequence data.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

from warpctc_pytorch import CTCLoss
from modules.wavenet import WaveNet
from modules.classifier import WaveNetClassifier


### Training parameters, etc.
num_iterations = 100000
dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


### construct wavenet and classifier models:
wavenet = WaveNet(256, 2, [(256, 256, 2, d) for d in dilations], 256, softmax=False)
classifier = WaveNetClassifier(None) # FIX
