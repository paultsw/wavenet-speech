"""
Basic CTC sanity check for decoder training.

We have a direct, one-to-one mapping between characters in input space X = {A,G,C,T} and target space Y = {a,g,c,t,|}.
We want to use the CTC loss function to train a decoder to collapse repeats between gaps (`|`) and preserve
repeats otherwise, e.g.:

    AAABAABBBA => (DECODE) => a|b|a|b|a

"""
import numpy as np
from scipy.ndimage.filters import generic_filter
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from modules.bytenet_decoder import ByteNetDecoder
from modules.rnn_decoder import RNNByteNetDecoder
from modules.sequence_decoders import argmax_decode
from warpctc_pytorch import CTCLoss
import traceback

### data/model/training parameters:
num_chars = 4
num_labels = num_chars+3 # + start, stop, pad
input_dim = 5
hidden_dim = 16
out_dim = 32
num_layers = 4
seq_length_low,seq_length_high = (40,50)
max_repeats = 3
max_timesteps = 200

### define random fixed embedding from target labels to input vectors:
label_map = {}
for k in range(1,num_chars):
    rand_vect = np.random.rand(input_dim)
    rand_vect.shape = (1,rand_vect.shape[0])
    label_map[k] = rand_vect

### data generator: repeat each target label a random number of times between 1-4:
def fetch_data():
    """
    Return randomly generated target sequence and sample randomly repeated input sequence.
    
    Outputs: (input_seq, label_seq, label_length), where:
    * input_seq: torch.FloatTensor Variable of shape `(1, input_dim, input_sequence_length)`,
    where `input_sequence_length` is randomly chosen by repeating each label a random number of times;
    * label_seq: torch.IntTensor Variable of shape `(1,label_sequence_length)` where
    `label_sequence_length` is chosen uniformly at random from `[seq_length_low, seq_length_high]`.
    * labe_length: an IntTensor Variable of shape `(1,)` containing the length of the label sequence.
    """
    # generate random labels:
    seq_length = np.random.randint(seq_length_low, seq_length_high+1)
    labels = np.random.randint(low=1, high=num_chars, size=(seq_length,))
    # randomly choose number of repeats:
    num_repeats = np.random.randint(1,high=max_repeats, size=(labels.shape[0],))
    repeated_labels = np.repeat(labels, num_repeats, axis=0)
    # generate input sequence:
    inputs = np.concatenate([label_map[k] for k in repeated_labels], axis=0)
    # convert to torch, unsqueeze (for bsz==1) and return:
    input_seq = torch.autograd.Variable(torch.from_numpy(inputs).float().unsqueeze(0).permute(0,2,1))
    label_seq = torch.autograd.Variable(torch.from_numpy(labels).int().unsqueeze(0))
    label_length = torch.autograd.Variable(torch.IntTensor([seq_length]))
    return (input_seq, label_seq, label_length)


### construct model, loss, and optimizer:
decoder = RNNByteNetDecoder(num_labels, input_dim, hidden_dim, out_dim, num_layers, max_timesteps)
#decoder = ByteNetDecoder(...)
loss_fn = CTCLoss()
opt = optim.Adamax(decoder.parameters(), lr=0.002)

### train against CTC loss:
def train():
    num_iterations = 2000000
    print_every = 5
    best_observed = 10000. # large value as default
    try:
        for k in range(num_iterations):
            opt.zero_grad()
            # fetch data:
            input_seq, label_seq, label_length = fetch_data()
            decoded, length = decoder.unfold(input_seq)
            transcription = decoded.cpu() # ~ seq, bsz, nlabels
            transcription_length = Variable(length, requires_grad=False).cpu()
            loss = loss_fn(transcription, label_seq, transcription_length, label_length)
            loss.backward()
            loss_scalar = loss.data[0]
            #avg_loss = loss / transcription.size(0)
            #avg_loss.backward()
            #loss_scalar = avg_loss.data[0]
            opt.step()
            if (k % print_every == 0):
                print("Step {0} | CTC: {1}".format(k, loss_scalar))
                best_observed = min(loss_scalar, best_observed)
                print("Ground Truth Labels:")
                print("".join([str(i) for i in label_seq.data[0].numpy().tolist()]))
                print("Predicted Labels:")
                transc_labels_list = argmax_decode(transcription.data.permute(1,0,2)).numpy()[0,:].tolist()
                print("".join([str(j) for j in transc_labels_list if j not in [0,5,6]]))
    except KeyboardInterrupt:
        print("Halted training from keyboard...")
    finally:
        print("Best observed loss: {}".format(best_observed))

if __name__ == '__main__':
    train()
