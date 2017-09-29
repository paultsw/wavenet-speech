"""
Entrypoint for training loops. Defines several functions to perform training, inference, etc. with
entry via command line and argparse.

Functions defined here can be imported (e.g. for hyperparameter optimization experiments).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

# custom modules:
from modules.wavenet import WaveNet
from modules.classifier import WaveNetClassifier as CTCNet
from models.raw_ctcnet import RawCTCNet
from modules.beam import Beam
from modules.sequence_decoders import argmax_decode, labels2strings
from utils.logging import Logger
from utils.config_tools import json_to_config, config_to_json
from utils.pore_model import PoreModelLoader
from utils.loaders import QueueLoader

# argument parsers:
import argparse
import os
import traceback

# flag to auto-detect whether or not CUDA is available
_CUDA_AVAIL_ = torch.cuda.is_available()


# ===== ===== Builders ===== =====
def build_loss_fns():
    xe_fn = nn.CrossEntropyLoss()
    ctc_fn = CTCLoss()
    return (xe_fn, ctc_fn)


def build_optimizer(settings, wavenet_params, ctcnet_params):
    """Construct optimizer."""
    # unpack and validate settings:
    optim_type = settings['type']
    assert (optim_type in ['adam', 'rmsprop', 'adagrad'])
    lr = optim_settings['lr']

    # construct AdaGrad optimizer:
    if optim_type == 'adagrad':
        opt = optim.Adagrad([{'params': wavenet_params}, {'params': ctcnet_params}],
                            learning_rate=lr)
    # construct RMSProp optimizer:
    if optim_type == 'rmsprop':
        raise NotImplementedError("RMSProp currently unsupported.")
    # construct Adam optimizer:
    if optim_type == 'adam':
        raise NotImplementedError("Adam currently unsupported.")

    return opt


def build_model(settings):
    """Build model from settings and return (wavenet_model, ctcnet model)"""
    # construct wavenet base model:
    wavenet_settings = settings['wavenet']
    in_dim_wn = wavenet_settings['in_dim']
    entry_kwidth_wn = wavenet_settings['entry_kwidth']
    assert (wavenet_settings['dilations'] or wavenet_settings['layers'])
    if (wavenet_settings['layers'] is not None):
        layers_wn = wavenet_settings['layers']
    else:
        layer_wn = [(in_dim_wn, in_dim_wn, 2, d) for d in wavenet_settings['dilations']]
    out_dim_wn = wavenet_settings['out_dim']
    wavenet_model = WaveNet(in_dim, entry_kwidth_wn, layers_wn, out_dim_wn, softmax=False)

    # construct CTCNet model:
    ctcnet_settings = settings['ctcnet']
    in_dim_ctc = ctcnet_settings['in_dim']
    num_labels = ctcnet_settings['num_labels']
    assert (ctcnet_settings['dilations'] or ctcnet_settings['layers'])
    if (ctcnet_settings['layers'] is not None):
        layers_ctc = ctcnet_settings['layers']
    else:
        layer_ctc = [(in_dim_ctc, in_dim_ctc, 2, d) for d in ctcnet_settings['dilations']]
    out_dim_ctc = ctcnet_settings['out_dim']
    downsample_rate = ctcnet_settings['pool_kernel_size']
    ctcnet_model = CTCNet(in_dim_ctc, num_labels, layers_ctc, out_dim_ctc,
                          pool_kernel_size=downsample_rate, input_kernel_size=2,
                          input_dilation=1, softmax=False)
    
    # return:
    if not _CUDA_AVAIL_:
        return (wavenet_model, ctcnet_model)
    else:
        return (wavenet_model.cuda(), ctcnet_model.cuda())


def build_hdf5_dataset(settings):
    """Construct dataset to read from an HDF5 file; returns a QueueLoader object."""
    assert (settings['type'] == 'hdf5')
    dataset_path = settings['dataset_path']
    nlevels = settings['num_signal_levels']
    nworkers = settings['num_workers']
    batch_size = settings['batch_size']
    sample_lengths = (settings['min_sample_length'], settings['max_sample_length'])
    max_iters = settings['max_iters']
    dataset = QueueLoader(dataset_path, num_signal_levels=nlevels, num_workers=nworkers, batch_size=batch_size,
                          sample_lengths=sample_lengths, max_iters=max_iters)
    return dataset


def build_pore_model_dataset(settings):
    """Constructs and returns an artificial pore model."""
    assert (settings['type'] == 'pore-model')
    max_iters = settings['max_iters']
    num_core_epochs = settings['num_core_epochs']
    num_ctc_epochs = settings['num_ctc_epochs']
    epoch_size = settings['epoch_size']
    batch_size = settings['batch_size']
    lengths = (settings['min_sample_length'], settings['max_sample_length'])
    pore_width = settings['pore_width']
    sample_rate = settings['sample_rate']
    noise = settings['noise']
    interleave_blanks = settings['interleave-blanks']
    # construct/return:
    return PoreModelLoader(max_iters, (num_core_epochs+num_ctc_epochs), epoch_size, batch_size,
                           lengths=pore_sample_lengths, pore_width=pore_width, sample_rate=sample_rate,
                           sample_noise=noise, interleave_blanks=interleave_blanks)


# ===== ===== Loss/train-step/decode functions ===== =====
def pretrain_core_wavenet(wavenet_, sig, xe_loss_fn, joint_optimizer):
    """Train the wavenet against itself in parallel."""
    joint_optimizer.zero_grad()
    pred_sig = wavenet_(sig[:,:,:-1])
    _, dense_sig = torch.max(sig[:,:,1:], dim=1)
    xe_loss = 0.
    for t in range(sig.size(2)-1):
        xe_loss = xe_loss + xe_loss_fn(pred_sig[:,:,t], dense_sig[:,t])
    avg_xe_loss = xe_loss / sig.size(2)
    avg_xe_loss.backward()
    joint_optimizer.step()
    return xe_loss.data[0]

def train_joint_loss(sig, seq, seq_lengths, wavenet_, ctcnet_, xe_loss_fn, ctc_loss_fn, joint_optimizer):
    """Train the wavenet & ctc-classifier jointly against both losses."""
    joint_optimizer.zero_grad()
    pred_sig = wavenet(sig)
    transcription = classifier(pred_sig)
    #-- cross entropy loss on wavenet output:
    _, dense_sig = torch.max(sig[:,:,1:], dim=1)
    xe_loss = 0.
    for t in range(sig.size(2)-1):
        xe_loss = xe_loss + xe_loss_fn(pred_sig[:,:,t], dense_sig[:,t])
    #-- ctc loss on predicted transcriptions
    probs = transcription.permute(2,0,1).contiguous() # expects (sequence, batch, logits)
    prob_lengths = Variable(torch.IntTensor([probs.size(0)] * batch_size))
    labels = seq.cpu().int() # expects flattened labels; label 0 is <BLANK>
    ctc_loss = ctc_loss_fn(probs, labels, prob_lengths, seq_lengths)
    if _CUDA_AVAIL_: ctc_loss = ctc_loss.cuda()
    #-- backprop:
    total_loss = xe_loss + ctc_loss
    average_loss = (xe_loss / sig.size(2)) + (ctc_loss / transcription.size(2))
    average_ctc_loss = (ctc_loss / transcription.size(2))
    average_loss.backward()
    #-- apply gradients and return loss values:
    joint_optimizer.step()
    return (total_loss.data[0], xe_loss_data[0], ctc_loss.data[0])


def train_ctc_loss(sig, seq, seq_lengths, wavenet_, ctcnet_, ctc_loss_fn, joint_optimizer):
    """Train the wavenet & ctc-classifier jointly against both losses."""
    joint_optimizer.zero_grad()
    pred_sig = wavenet(sig)
    transcription = classifier(pred_sig)
    #-- ctc loss on predicted transcriptions
    probs = transcription.permute(2,0,1).contiguous() # expects (sequence, batch, logits)
    prob_lengths = Variable(torch.IntTensor([probs.size(0)] * batch_size))
    labels = seq.cpu().int() # expects flattened labels; label 0 is <BLANK>
    ctc_loss = ctc_loss_fn(probs, labels, prob_lengths, seq_lengths)
    if _CUDA_AVAIL_: ctc_loss = ctc_loss.cuda()
    #-- backprop:
    average_ctc_loss = (ctc_loss / transcription.size(2))
    average_ctc_loss.backward()
    #-- apply gradients and return loss values:
    joint_optimizer.step()
    return (total_loss.data[0], xe_loss_data[0], ctc_loss.data[0])


def decode(logits, options):
    """
    Perform either beam search decoding or argmax decoding over a
    sequence of logits; return a batch of decoded nucleotide sequences.
    """
    # unpack settings from dict:
    mode = options['mode']
    assert (mode in ['argmax', 'beam'])
    lookup_dict = options['lookup'] or { 0: '', 1: 'A', 2: 'G', 3: 'C', 4: 'T' }

    # in argmax decoding mode:
    if (mode == 'argmax'):
        return labels2strings(argmax_decode(logits))
    
    # in beam search decoding mode:
    _batch = options['batch_size']
    _nlabels = options['num_labels']
    decoder = BeamSearchDecoder(_batch, _nlabels, beam_width=options['beamwidth'], cuda=options['cuda'])
    decoded_sequence = decoder.decode(logits)
    return labels2strings(decoded_sequences)


# ===== ===== Main training/evaluation functions ===== =====
def evaluate_loop(params):
    """[TBD: copy from IPyNB; build decoder]"""
    pass


def train_loop(cfg):
    """Train a model for some number of iterations."""
    # unpack settings from config:
    optim_cfg = cfg['optim']
    run_dir = cfg['run_dir']
    dataset_cfg = cfg['dataset']

    # build logger:
    logger = Logger(run_dir)
    
    # build models:
    wavenet, ctcnet = build_models()
    
    # build optimizer:
    optimizer = build_optimizer(optim_cfg, wavenet.parameters(), ctcnet.parameters())

    # build loss functions:
    xe_loss_fn, ctc_loss_fn = build_loss_fns()

    # build dataset:
    assert (dataset_cfg['type'] in ['hdf5', 'pore-model'])
    if dataset_cfg['type'] == 'hdf5':
        dataset = build_hdf5_dataset(dataset_cfg)
    else:
        dataset = build_pore_model_dataset(dataset_cfg)

    # perform training:
    for k in range(cfg['num_iterations']):
        # [[train here]]


# ===== ===== Read configuration file and run if called from command line ===== =====
if __name__ == "__main__":
    ### Temporary ad-hoc configuration parser:
    json_parser = argparse.ArgumentParser(description="Run WaveNet-CTC model.")
    json_parser.add_argument("task", choices=('train','evaluate'))
    json_parser.add_argument("config_path")
    args = json_parser.parse_args()
    config = json_to_config(args.config_path)
    if args.task == "train": train(config['train_settings'])
    if args.task == "evaluate": evaluate(config['eval_settings'])
    
    
    """ Below: detailed parser, work in progress
    ### Top-level parser and common flags:
    top_parser = argparse.ArgumentParser(description="Run WaveNet-CTC model.")
    top_parser.add_argument("--data", choices=('pore-model', 'hdf5'), default='hdf5')
    top_parser.add_argument("--decode", choices=('argmax', 'beam'), default='beam')
    top_parser.add_argument("--loss", choices=('avg-ctc','avg-tot','tot','ctc'), default='avg-tot')
    top_parser.add_argument("--wavenet-path", help="Path to saved wavenet-base model to restore from")
    top_parser.add_argument("--classifier-path", help="Path to saved classifier model to restore from")
    subparsers = top_parser.add_subparsers(help="command", dest='command')
    subparsers.required = True
    
    ### Training sub-parser:
    train_parser = subparsers.add_parser("train", parents=['top_parser'])
    train_parser.add_argument("--epochs")
    train_parser.add_argument("--optimizer")
    train_parser.add_argument("--log-every", type=int)
    train_parser.add_argument("--save-every", type=int)

    ### Evaluation sub-parser:
    eval_parser = subparsers.add_parser("evaluate", parents=['top_parser'])
    eval_parser.add_argument("--blah")
    """
