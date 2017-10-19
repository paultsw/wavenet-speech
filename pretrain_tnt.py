"""
Pretraining on raw signal model using an Encoder-Decoder architecture.

Since an infinite amount of data is sampled from a generative model, no validation is performed.

Usage:
* If '--visdom' is added, make sure that Visdom Server is already running before running this script:
  $ python -c 'visdom.server'
* Then, run this file:
  $ python pretrain_tnt.py
* Alternatively, do both of the above via:
  $ python -c 'visdom.server' && python pretrain_tnt.py
"""
# numerical libs:
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from warpctc_pytorch import CTCLoss
# custom modules/datasets:
from modules.raw_ctcnet import RawCTCNet
from modules.bytenet_decoder import ByteNetDecoder
from modules.sequence_decoders import argmax_decode, BeamSearchDecoder, labels2strings
from utils.raw_signal_generator import RawSignalGenerator
# torchnet imports:
import torchnet as tnt
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger, VisdomTextLogger
# etc.
from tqdm import tqdm
import argparse
import itertools

### helper functions to reshape to/from model<->loss.
# N.B.: Both of these functions onnly accepts `Variable` type so that gradients are preserved.
def to_concat(labels_batch, lengths):
    """Concatenate together, excluding the padding values."""
    concat_seqs = []
    for k in range(len(labels_batch)):
        concat_seqs.append(labels_batch[k,0:lengths.data[k]])
    return torch.cat(concat_seqs, 0)

def to_stack(labels_concat, lengths):
    """Stack together (B-S dimensions), with padding values at the end of the sequence."""
    curr = 0
    max_length = max(lengths.data)
    stack_seqs = []
    for k in range(len(lengths)):
        seq = labels_concat[curr:(curr+lengths.data[k])]
        pad_amt = max_length-lengths.data[k]
        if pad_amt > 0:
            zero_pad_var = Variable(torch.zeros(pad_amt).long())
            padded_seq = torch.cat((seq, zero_pad_var),dim=0)
        else:
            padded_seq = seq
        stack_seqs.append(padded_seq)
        curr += lengths.data[k]
    return torch.stack(stack_seqs, dim=0)


### helper class for data-loading
class RawSignalDataset(object):
    """Wrap RawSignalGenerator in an Iterator."""
    def __init__(self, cfg):
        self._niterations = cfg['epoch_size']
        self.kmer_model = "./utils/r9.4_450bps.5mer.template.npz"
        self.reference_hdf_path = "./utils/r9.4_450bps.5mer.ecoli.model/reference.hdf5"
        self.read_length_model = (90,100) # or "./utils/r9.4_450bps.5mer.ecoli.model/read_lengths.npy"
        self.sample_rate = 800.
        self.batch_size = cfg['batch_size']
        self._dataset = RawSignalGenerator(self.kmer_model, self.reference_hdf_path, self.read_length_model,
                                           self.sample_rate, self.batch_size, dura_shape=None, dura_rate=None)
        self._ctr = 0

    def __len__(self):
        return self._niterations
        
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("CustomRange index out of range")
        return self._dataset.fetch() # (signals,sequences,signal_lengths,sequence_lengths) ~ LongTensor Variables, on CPU


### main training loop; use visdom logging:
def main(cfg):
    #-- construct dataset loader:
    def get_iterator():
        return RawSignalDataset(cfg)
    
    #-- construct model:
    num_features = 512
    feature_kwidth = 3
    encoder_dim = 512
    enc_layers = [(512,512,d,2) for d in [1,2,4,8,16]] * 5 + [(512,512,d,3) for d in [1,2,4,8,16]] * 5
    enc_out_dim = 512
    num_labels = 5
    dec_channels = 512
    dec_kwidth = 3
    dec_out_dim = 256
    dec_layers = [(3,d) for d in [1,2,4,8,16]] * 5
    encoder = RawCTCNet(num_features, feature_kwidth, encoder_dim, enc_layers, enc_out_dim,
                        input_kernel_size=2, input_dilation=1, positions=False, softmax=False, causal=False)
    #decoder = ByteNetDecoder(num_labels, encoder_dim, dec_channels, dec_kwidth, dec_out_dim, dec_layers, block='mult')
    print("Constructed model.")

    #-- loss function & computation:
    ctc_loss_fn = CTCLoss()
    print("Constructed loss function.")
    def model_loss(sample):
        # unpack inputs:
        signals, sequences, signal_lengths, sequence_lengths = sample
        """ # TODO: fix decoder to allow for arbitrary sequential timesteps
        # get encodings from RawCTCNet:
        encoded_seq = encoder(signals.unsqueeze(1))
        # compute next-step prediction via bytenet decoder:
        next_timestep_preds = decoder(sequences[:,0:-1], encoded_seq[:,:,0:(sequences.size(1)-1)]) # [ISSUE HERE: size constraint]
        """
        next_timestep_preds = encoder(signals.unsqueeze(1))
        # compute CTC loss and return:
        transcriptions = next_timestep_preds.permute(2,0,1) # [B,C,S=>S,B,C]
        transcription_lengths = Variable(torch.IntTensor([transcriptions.size(0)] * transcriptions.size(1)))
        label_lengths = sequence_lengths.int()
        labels = to_concat(sequences, label_lengths).int()
        #transcription_lengths = (signal_lengths.int() - 1)
        #label_lengths = (sequence_lengths.int() - 1)
        #labels = to_concat(sequences[1:signals.size(1)+1], label_lengths)
        loss = ctc_loss_fn(transcriptions, labels, transcription_lengths, label_lengths)
        return loss, transcriptions
    
    #-- optimizer:
    #opt = optim.Adamax([{"params": encoder.parameters()},
    #                    {"params": decoder.parameters()}],
    #                   lr=0.002)
    opt = optim.Adamax(encoder.parameters(), lr=0.002)
    print("Constructed optimizer.")
    
    #-- beam search:
    beam_search = BeamSearchDecoder(cfg['batch_size'], num_labels, beam_width=6)
    print("Constructed beam search decoder.")

    #-- engine, meters, loggers:
    engine = Engine()
    loss_meter = tnt.meter.MovingAverageValueMeter(windowsize=5)
    #train_loss_logger = VisdomPlotLogger('line', opts={ 'title': 'Train Loss' })
    #beam_decode_logger = VisdomTextLogger(update_type='APPEND')
    print("Constructed engine. Running training loop...")

    #-- reset meters:
    def reset_all_meters():
        """Reset all meters."""
        loss_meter.reset()

    #-- sample hook:
    def on_sample(state):
        """What to do after each sample (after obtaining a data sample)."""
        pass # [Don't do anything for now; we want to preprocess here in the future.]

    #-- on update:
    def on_update(state):
        """What to do after each SGD update. [Training only.]"""
        pass # [Don't do anything for now.]

    #-- forward pass hook:
    def on_forward(state):
        """Update loggers at each forward pass. [Testing only.]"""
        loss_meter.add(state['loss'].data[0])

    #-- start epoch hook:
    def on_start_epoch(state):
        """At the start of each epoch. [Training only.]"""
        reset_all_meters()
        state['iterator'] = tqdm(state['iterator'])

    #-- end epoch hook:
    def on_end_epoch(state):
        """After each epoch, perform validation and do a beam search decoding printout. [Training only.]"""
        # Log training info to loggers:
        # [TODO: log info to training loggers here]
        # [TODO: run argmax-/beam-decoding here]
        # [TODO: save models here]
        reset_all_meters()
    
    #-- set up engine:
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(model_loss, get_iterator(), maxepoch=cfg['max_epochs'], optimizer=opt)


if __name__ == '__main__':
    # read config and run main():
    parser = argparse.ArgumentParser(description="Train an encoder-decoder model on realistically-modelled data.")
    parser.add_argument("--max_epochs", dest='max_epochs', default=100, help="Number of epochs")
    parser.add_argument("--epoch_size", dest='epoch_size', default=10000, help="Number of steps per epoch")
    parser.add_argument("--batch_size", dest='batch_size', default=8, help="Number of sequences per batch")
    args = parser.parse_args()
    cfg = {
        'max_epochs': args.max_epochs,
        'epoch_size': args.epoch_size,
        'batch_size': args.batch_size,
    }
    main(cfg)
