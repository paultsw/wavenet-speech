"""
Quick evaluation of raw CTC network on Gaussian 5mer model.
"""
# import the usual suspects
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from warpctc_pytorch import CTCLoss

# import gaussian model, RawCTCNet, sequential decoder:
from utils.gaussian_kmer_model import RawGaussianModelLoader
from modules.raw_ctcnet import RawCTCNet
from modules.sequence_decoders import argmax_decode, labels2strings, BeamSearchDecoder

# create artificial data model:
max_iterations = 1000000 # 1 million examples
num_epochs = 100
epoch_size = 10000
kmer_model_path = "utils/r9.4_450bps.5mer.template.npz"
batch_size = 8
upsample_rate = 3
min_sample_len = 30
max_sample_len = 40
dataset = RawGaussianModelLoader(max_iterations, num_epochs, epoch_size, kmer_model_path, batch_size=batch_size,
                                 upsampling=upsample_rate, lengths=(min_sample_len,max_sample_len))

nfeats = 512
feature_kwidth = 1
num_labels = 5
dilations = [1, 2, 4, 8, 16, 32, 64,
             1, 2, 4, 8, 16, 32, 64,
             1, 2, 4, 8, 16, 32, 64]
layers = [(nfeats, nfeats, 2, d) for d in dilations] + [(nfeats, nfeats, 3, d) for d in dilations]
out_dim = 512
ctcnet = RawCTCNet(nfeats, feature_kwidth, num_labels, layers, out_dim, input_kernel_size=2, input_dilation=1,
                   softmax=False, causal=True)
batch_norm = torch.nn.BatchNorm1d(1)

# automatically place on GPU if detected:
if torch.cuda.is_available():
    print("CUDA device detected. Placing on GPU device 0.")
    ctcnet.cuda()

ctc_loss_fn = CTCLoss()
ctc_opt = optim.Adam(ctcnet.parameters(), weight_decay=0.0001)
#ctc_opt = optim.Adagrad(ctcnet.parameters(), lr=0.00001)

# run this IPyNB cell multiple times (as often as necessary until it converges)
num_iterations = 1000
log_every = 10
for k in range(num_iterations):
    ctc_opt.zero_grad()
    signals, sequences, sequence_lengths = dataset.fetch()
    probas = ctcnet(batch_norm(signals.unsqueeze(1)))
    transcriptions = probas.permute(2,0,1).cpu() # need seq x batch x dim
    transcription_lengths = Variable(torch.IntTensor([transcriptions.size(0)] * batch_size))
    ctc_loss = ctc_loss_fn(transcriptions, sequences, transcription_lengths, sequence_lengths)
    avg_ctc_loss = (ctc_loss / transcriptions.size(0))
    avg_ctc_loss.backward()
    if (k % log_every == 0):
        print("Loss @ step {0}: {1} | Per-Logit: {2}".format(k, ctc_loss.data[0], avg_ctc_loss.data[0]))
    ctc_opt.step()
