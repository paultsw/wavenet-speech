"""
Attempt to decode a set of kmers into a nucleotide sequence using the ByteNet decoder.

Results: the ByteNetDecoder module is able to decode the nucleotide sequence extremely quickly
using Cross Entropy loss. (Experiment with CTC loss currently unperformed but planned.)
"""
import numpy as np
from scipy.ndimage.filters import generic_filter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from modules.bytenet_decoder import ByteNetDecoder
from warpctc_pytorch import CTCLoss


### Helper functions to convert between concatenated representation and stacked representation:
# (Both of these only work for tensors, not variables.)
def to_concat(labels_batch, lengths):
    """Concatenate together, excluding the padding values."""
    concat_seqs = []
    for k in range(len(labels_batch)):
        concat_seqs.append(labels_batch[k,0:lengths[k]])
    return torch.cat(concat_seqs, 0)

def to_stack(labels_concat, lengths):
    """Stack together (B-S dimensions), with padding values at the end of the sequence."""
    curr = 0
    max_length = max(lengths)
    stack_seqs = []
    for k in range(len(lengths)):
        seq = labels_concat[curr:(curr+lengths[k])]
        padded_seq = torch.cat((seq, torch.zeros(max_length-lengths[k]).long()),dim=0)
        stack_seqs.append(padded_seq)
        curr += lengths[k]
    return torch.stack(stack_seqs, dim=0)

### combine a bunch of numpy sequences into a batch (assumes equal sequence lengths on all sequences):
def batchify(seqs_list):
    return np.stack(seqs_list, axis=0)

### Construct function to convert random nucleotide sequences to kmer labels:
batch_size = 32
nt_length = 100
nts_to_5mer_map = lambda nt: np.sum((nt-np.ones(nt.shape)) * np.array([4**4, 4**3, 4**2, 4**1, 4**0]))
def fetch_stitch_nts():
    nts_list =  []
    kmers_list = []
    for b in range(batch_size):
        # generate a random nucleotide sequence:
        nts = np.random.randint(1,high=5,size=(nt_length,), dtype=np.int64)
        # stitch into 5mers:
        kmers = generic_filter(nts, nts_to_5mer_map, size=(5,), mode='constant')
        kmers = kmers[2:-2].astype(int) # remove padding values
        nts_list.append(nts)
        kmers_list.append(kmers)
    # stack into batch and return:
    return torch.from_numpy(batchify(nts_list)).long(), torch.from_numpy(batchify(kmers_list)).long()


### Main training loop, using cross entropy loss:
def main_ce():
    print("Constructing model and optimizers...")
    ### Construct bytenet decoder:
    num_labels = 5
    encoding_dim = 1024
    channels = 256
    kwidth = 3
    output_dim = 512
    layers = [(3,1), (3,2), (3,4), (3,8), (3,16)]
    kmer_embedding = nn.Embedding(1024,encoding_dim)
    decoder = ByteNetDecoder(num_labels, encoding_dim, channels, kwidth, output_dim, layers, block='mult')

    ### construct loss & optimizer:
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adamax(decoder.parameters(), lr=0.002)

    ### training loop:
    num_iterations = 2000000 # 2 million steps
    print_every = 5
    print("Running training loop for {} iterations... Press Ctrl-C to end.".format(num_iterations))
    try:
        best_observed = 1000000. # arbitrarily high value
        for k in range(num_iterations):
            opt.zero_grad()
            # generate data/prediction:
            nt_sequence, kmer_sequence = fetch_stitch_nts()
            nt_var = Variable(nt_sequence)
            kmer_var = Variable(kmer_sequence)
            embedded_kmers = kmer_embedding(kmer_var).transpose(1,2) # BS => BCS
            pred_nts = decoder(nt_var[:,0:embedded_kmers.size(2)], embedded_kmers)
            # compute loss w/r/t next-timestep value:
            loss = 0.
            for t in range(pred_nts.size(2)):
                loss = loss + loss_fn(pred_nts[:,:,t], nt_var[:,t+1])
            # backprop:
            loss.backward()
            opt.step()
            # log if scheduled:
            if (k % print_every == 0):
                loss_scalar = loss.data[0]
                avg_loss_scalar = loss_scalar / pred_nts.size(0)
                if avg_loss_scalar < best_observed: best_observed = avg_loss_scalar
                print("Step {0} | Loss: {1} | Avg Loss: {2}".format(k, loss_scalar, avg_loss_scalar))
    except KeyboardInterrupt:
        print("Halted training from keyboard.")
    finally:
        print("Best observed loss: {}".format(best_observed))


### Main training loop, using CTC loss:
def main_ctc():
    ### Construct bytenet decoder:
    num_labels = 5
    encoding_dim = 1024
    channels = 256
    kwidth = 3
    output_dim = 512
    layers = [(3,1), (3,2), (3,4), (3,8), (3,16)]
    kmer_embedding = nn.Embedding(1024,encoding_dim)
    decoder = ByteNetDecoder(num_labels, encoding_dim, channels, kwidth, output_dim, layers, block='mult')

    ### construct loss & optimizer:
    loss_fn = CTCLoss()
    opt = optim.Adamax(decoder.parameters(), lr=0.002)

    ### training loop: # [TODO: FIX THIS TO USE CTCLOSS-STYLE VARIABLE SHAPES]
    num_iterations = 2000000 # 2 million steps
    print_every = 50
    try:
        best_observed = 1000000. # arbitrarily high value
        for k in range(num_iterations):
            opt.zero_grad()
            # generate data/prediction:
            nt_sequence, kmer_sequence = fetch_stitched_nts()
            embedded_kmers = kmer_embedding(kmer_sequence).transpose(1,2)
            pred_nts = decoder(nt_sequence[0:-1], embedded_kmers) # [FIX]
            # compute loss:
            loss = 0.
            for k in range(pred_nts.size(2)):
                loss = loss + loss_fn(pred_nts[:,:,k], nts_sequence[:,k])
            # backprop:
            loss.backward()
            opt.step()
            # log if scheduled:
            if (k % print_every == 0):
                loss_scalar = loss.data[0]
                avg_loss_scalar = loss_scalar / pred_nts.size(0)
                if avg_loss_scalar < best_observed: best_observed = avg_loss_scalar
                print("Step {0} | Loss: {1} | Avg Loss: {2}".format(k, loss_scalar, avg_loss))
    except KeyboardInterrupt:
        print("Halted training from keyboard.")
    finally:
        print("Best observed loss: {}".format(best_observed))


### read input arg and figure out which training loop to run:
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Test stitching ability")
    parser.add_argument("--loss", dest='loss', choices=('ce','ctc'), default='ce', help="Which loss to use")
    args = parser.parse_args()
    if args.loss == 'ce':
        main_ce()
    else:
        main_ctc()
