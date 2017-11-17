"""
Pretraining on raw signal model using an Encoder-Decoder architecture.

This version uses a bytenet-style RNN as the decoder.

Since an infinite amount of data is sampled from a generative model, no validation is performed;
upon finishing each epoch, the following is done:
* beam search decoder [#beams==4] used to print example output on example input;
* model is auto-saved to a specified run directory;
* learning rate decreased if no improvement in loss beyond some threshold.
"""
# numerical libs:
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from warpctc_pytorch import CTCLoss
# custom modules/datasets:
from modules.raw_ctcnet import RawCTCNet
from modules.rnn_decoder import RNNByteNetDecoder
from modules.sequence_decoders import argmax_decode, BeamSearchDecoder, labels2strings
from utils.raw_signal_generator import RawSignalGenerator
# torchnet imports:
import torchnet as tnt
from torchnet.engine import Engine
# etc.
from tqdm import tqdm
import argparse
import itertools
import os

### helper functions to reshape to/from model<->loss.
# N.B.: Both of these functions only accepts `Variable` type so that gradients are preserved.
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
            zero_pad_var = Variable(torch.zeros(pad_amt).long(), requires_grads=False)
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
        self.read_length_model = (20,30) # or "./utils/r9.4_450bps.5mer.ecoli.model/read_lengths.npy"
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
def main(cfg, cuda=torch.cuda.is_available()):
    #-- construct dataset loader:
    def get_iterator():
        return RawSignalDataset(cfg)
    
    # Construct encoder-decoder architecture:
    print("Constructing encoder-decoder model...")
    num_features = 512
    feature_kwidth = 3
    encoder_dim = 512
    enc_layers = [(512,512,d,2) for d in [1,2,4,8,16]] * 5
    enc_out_dim = 512
    num_labels = 7
    dec_hdim = 512
    dec_out_dim = 1024
    dec_layers = 5
    max_time = 100
    encoder = RawCTCNet(num_features, feature_kwidth, encoder_dim, enc_layers, enc_out_dim,
                        input_kernel_size=2, input_dilation=1, positions=False, softmax=False, causal=False)
    decoder = RNNByteNetDecoder(num_labels, encoder_dim, dec_hdim, dec_out_dim, dec_layers,
                                pad=0, start=5, stop=6, max_timesteps=max_time)
    print("Constructed model.")
    if cuda:
        encoder.cuda()
        decoder.cuda()
        print("CUDA detected... Placed encoder and decoder on GPU memory.")

    #-- loss function & computation:
    ctc_loss_fn = CTCLoss()
    print("Constructed loss function.")
    def model_loss(sample):
        # unpack inputs:
        signals, sequences, signal_lengths, sequence_lengths = sample
        if cuda: signals = signals.cuda()
        # get encodings from RawCTCNet:
        encoded_seq = encoder(signals.unsqueeze(1))
        # decode sequence into nucleotides via bytenet decoder:
        decoded_seq, decoded_lengths = decoder.unfold(encoded_seq)
        # compute CTC loss and return:
        transcriptions = decoded_seq.cpu()
        transcription_lengths = Variable(decoded_lengths, requires_grad=False).cpu()
        label_lengths = sequence_lengths.int().cpu()
        labels = to_concat(sequences, label_lengths).int().cpu()
        loss = ctc_loss_fn(transcriptions, labels, transcription_lengths, label_lengths)
        #avg_loss = loss / transcriptions.size(0)
        return loss, transcriptions
    
    #-- optimizer:
    opt = optim.Adadelta([{'params': encoder.parameters(), 'lr': 0.0001},
                          {'params': decoder.parameters(), 'lr': 0.0001}])
    print("Constructed optimizer.")

    # -- scheduler:
    sched = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.8, patience=10,
                                           verbose=False, threshold=1e-4, threshold_mode='rel')
    print("Constructed LR scheduler (Plateau)")

    #-- beam search: [TODO: fix this to make START/STOP/PAD optional]
    beam_search = BeamSearchDecoder(cfg['batch_size'], num_labels, beam_width=4, cap_seqs=False, cuda=cuda)
    print("Constructed beam search decoder.")

    #-- engine, meters, loggers:
    engine = Engine()
    loss_meter = tnt.meter.MovingAverageValueMeter(windowsize=5)
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
        if (state['t'] % cfg['print_every'] == 0):
            tqdm.write("Step: {0} | Loss: {1}".format(state['t'], state['loss'].data[0]))

    #-- start epoch hook:
    def on_start_epoch(state):
        """At the start of each epoch. [Training only.]"""
        reset_all_meters()
        state['iterator'] = tqdm(state['iterator'])

    #-- end epoch hook:
    def on_end_epoch(state):
        """
        After each epoch, perform validation on 10 samples;
        print beam search out; (maybe) reduce LR. [Training only.]
        """
        ### perform 10 steps of validation and average their losses:
        val_losses = []
        base_seqs = []
        val_data_iter = get_iterator()
        for _ in range(10):
            # get data:
            val_sample = val_data_iter[0]
            val_loss, transcriptions = model_loss(val_sample)
            val_losses.append(val_loss.data[0])
            sequences = val_sample[1]
            base_seqs.append( (sequences, transcriptions) )
        avg_val_loss = sum(val_losses) / 10

        ### send average validation loss to LR scheduler:
        sched.step(avg_val_loss)

        ### perform argmax decoding over true/called seqs:
        _nt_dict_ = {0: '', 1: 'A', 2: 'G', 3: 'C', 4: 'T', 5: '<s>', 6: '</s>'}
        for true_seqs, transcriptions in base_seqs:
            true_nts = labels2strings(true_seqs, lookup=_nt_dict_)
            reshaped_tscts = transcriptions.permute(1,2,0) # => (bsz,id,seq)
            _, pred_seqs = beam_search.decode(reshaped_tscts)
            pred_nts = ["".join([_nt_dict_[lbl] for lbl in hyp]) for hyp in pred_seqs]
            for i in range(min(len(true_nts), len(pred_nts))):
                tqdm.write("True Seq: {0}".format(true_nts[i]))
                tqdm.write("Pred Seq: {0}".format(pred_nts[i]))

        ### save model:
        try:
            enc_path = os.path.join(cfg['save_dir'], "cnn_encoder.epoch_{0}.pth".format(state['epoch']))
            dec_path = os.path.join(cfg['save_dir'], "rnn_decoder.epoch_{0}.pth".format(state['epoch']))
            torch.save(encoder.state_dict(), enc_path)
            torch.save(decoder.state_dict(), dec_path)
            tqdm.write("Saved encoder-decoder models.")
        except:
            tqdm.write("Unable to serialize models. Moving on...")

        ### reset meters for next epoch:
        reset_all_meters()

    #-- set up engine and start training:
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(model_loss, get_iterator(), maxepoch=cfg['max_epochs'], optimizer=opt)


if __name__ == '__main__':
    # read config and run main():
    parser = argparse.ArgumentParser(description="Train an encoder-decoder model on realistically-modelled data.")
    parser.add_argument("--max_epochs", dest='max_epochs', type=int, default=100, help="Number of epochs [100]")
    parser.add_argument("--epoch_size", dest='epoch_size', type=int, default=10000, help="Number of steps per epoch [10K]")
    parser.add_argument("--print_every", dest='print_every', type=int, default=25, help="Log the loss to stdout every N steps [25]")
    parser.add_argument("--batch_size", dest='batch_size', type=int, default=8, help="Number of sequences per batch [8]")
    parser.add_argument("--save_dir", dest='save_dir', default="./", help="Path to save models at each epoch [pwd]")
    args = parser.parse_args()
    cfg = {
        'max_epochs': args.max_epochs,
        'epoch_size': args.epoch_size,
        'batch_size': args.batch_size,
        'print_every': args.print_every,
        'save_dir': args.save_dir
    }
    main(cfg)
