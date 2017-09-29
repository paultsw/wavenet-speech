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
from utils.loaders import QueueLoader
from utils.logging import Logger
from utils.config_tools import json_to_config, config_to_json

# argument parsers:
import argparse
import os
import traceback

CUDA_AVAIL = torch.cuda.is_available()

def train_step(wavenet, ctcnet, xe_loss_fn, ctc_loss_fn, sig, seq, lengths, opt, batch_size, averaged=True):
    """Take one training step of WaveNet."""
    # 0. clear gradients:
    opt.zero_grad()
    
    # 1. run wavenet on signal:
    wavenet_pred = wavenet(sig[:,:,0:-1])

    # 2. run classifier on wavenet output distribution:
    transcription = ctcnet(wavenet_pred)

    # 3. compute NLL between true signal and wavenet output:
    _, dense_sig = torch.max(sig[:,:,1:], dim=1)
    xe_loss = 0.
    for t in range(sig.size(2)-1):
        xe_loss = xe_loss + xe_loss_fn(wavenet_pred[:,:,t], dense_sig[:,t])

    # 4. compute CTC loss between true seq and predicted seq:
    probs = transcription.permute(2,0,1).contiguous() # ~ (seq, batch, logits)
    prob_lengths = Variable(torch.IntTensor([probs.size(0)] * batch_size))
    labels = seq.cpu().int()
    labels = labels + Variable(torch.ones(seq[0].size()).int()) # <0> == <BLANK>
    ctc_loss = ctc_loss_fn(probs, labels, prob_lengths, lengths)
    if CUDA_AVAIL: ctc_loss = ctc_loss.cuda()
    
    # 5. backprop:
    total_joint_loss = ctc_loss + xe_loss
    avg_xe_loss = xe_loss / sig.size(2)
    avg_ctc_loss = ctc_loss / transcription.size(2)
    avg_joint_loss = avg_xe_loss + avg_ctc_loss
    avg_joint_loss.backward()
    opt.step()

    # 6. return loss values:
    if not averaged:
        return (xe_loss.data[0], ctc_loss.data[0], total_joint_loss.data[0])
    else:
        return (avg_xe_loss.data[0], avg_ctc_loss.data[0], avg_joint_loss.data[0])


def build_optimizer(optim_cfg, wave_params, ctc_params):
    """Build an optimizer based on config settings."""
    optim_type = optim_cfg['type']
    lr = optim_cfg['learning_rate']
    wd = optim_cfg['wd']
    max_grad_norm = optim_cfg['max_grad_norm']
    assert (optim_type in ['adam', 'rmsprop'])
    if optim_type == 'adam':
        return optim.Adam([{'params': wave_params}, {'params': ctc_params}],
                          lr=lr, weight_decay=wd)
    if optim_type == 'rmsprop':
        return optim.RMSprop(params, lr=lr, weight_decay=wd, centered=False)


def train(config):
    """
    Main training loop.
    """
    ###===== construct wavenet:
    # base wavenet settings:
    signal_dim = config['model']['base']['signal_dim']
    entry_kwidth = config['model']['base']['entry_kwidth']
    wavenet_kwidth = config['model']['base']['kernel']
    wavenet_dilations = config['model']['base']['dilations']
    wavenet_layers = [(signal_dim, signal_dim, wavenet_kwidth, d) for d in wavenet_dilations]
    # classifier settings:
    num_labels = config['model']['classifier']['num_labels']
    classifier_kwidth = config['model']['classifier']['kernel']
    classifier_dilations = config['model']['classifier']['dilations']
    classifier_layers = [(signal_dim, signal_dim, classifier_kwidth, d) for d in classifier_dilations]
    downsample = config['model']['classifier']['downsample']
    out_dim = config['model']['classifier']['out_dim']
    wavenet_core = WaveNet(signal_dim, entry_kwidth, wavenet_layers, signal_dim, softmax=False)
    wavenet_class = WaveNetClassifier(signal_dim, num_labels, classifier_layers, out_dim,
                                      pool_kernel_size=downsample, input_kernel_size=2,
                                      input_dilation=1, softmax=False)

    ###===== optionally restore weights to continue training:
    restore_wavenet_path = config['training']['restore_wavenet_path']
    restore_classifier_path = config['training']['restore_classifier_path']
    if restore_wavenet_path is not None:
        wavenet_core.load_state_dict(
            torch.load(restore_wavenet_path, map_location=lambda storage, loc: storage))
    if restore_classifier_path is not None:
        wavenet_class.load_state_dict(
            torch.load(restore_classifier_path, map_location=lambda storage, loc: storage))
    if CUDA_AVAIL:
        wavenet_core.cuda()
        wavenet_class.cuda()

    ###===== construct loss functions:
    xe_loss_fn = nn.CrossEntropyLoss()
    ctc_loss_fn = CTCLoss()

    ###===== construct optimizers:
    optim_base_cfg = config['training']['optim']['base']
    optim_classifier_cfg = config['training']['optim']['classifier']
    opt = build_optimizer(optim_classifier_cfg, wavenet_core.parameters(), wavenet_class.parameters())

    ###===== construct data loader and logging object:
    print("Constructing data queue...")
    train_dataset_path = config['training']['training_data']
    validation_dataset_path = config['training']['validation_data']
    save_dir = config['training']['save_dir']
    num_epochs = config['training']['num_epochs']
    max_iters = config['training']['max_iters']
    nworkers = config['training']['nworkers']
    batch_size = config['training']['batch_size']
    sample_lengths = (config['training']['min_sample_length'], config['training']['max_sample_length'])
    train_dataset = QueueLoader(train_dataset_path, num_signal_levels=signal_dim, num_workers=nworkers,
                                batch_size=batch_size, sample_lengths=sample_lengths, max_iters=max_iters)
    if CUDA_AVAIL: train_dataset.cuda()
    if validation_dataset_path is not None:
        validation_dataset = QueueLoader(validation_dataset_path)
    print("...Done.")
    logger = Logger(save_dir)

    ###===== run training loop:
    print_every = config['training']['print_every']
    save_every = config['training']['save_every']
    timestep = 0
    try:
        for _ in range(max_iters):
            signals, seqs, sig_lengths, seq_lengths = train_dataset.dequeue()
            xe_loss, ctc_loss, joint_loss = train_step(
                wavenet_core, wavenet_class, xe_loss_fn, ctc_loss_fn, signals, seqs, seq_lengths, opt, batch_size, averaged=True)
            if timestep % print_every == 0:
                # [TODO: enable validation set]
                #valid_signal, valid_sequence = dataset.fetch_validation_data()
                #valid_nll_loss, valid_ctc_loss, valid_joint_loss = wavenet_eval_step(
                #    wavenet_core, wavenet_class, nll_loss_fn, ctc_loss_fn, valid_signal, valid_sequence)
                valid_xe_loss = valid_ctc_loss = valid_joint_loss = 0.
                logger.log("XE", timestep, xe_loss, valid_xe_loss)
                logger.log("CTC", timestep, ctc_loss, valid_ctc_loss)
                logger.log("SUM", timestep, joint_loss, valid_joint_loss)
            if timestep % save_every == 0: logger.save(timestep, wavenet_core, wavenet_class)
            if timestep == max_iters: break
            timestep += 1
    except KeyboardInterrupt:
        print("-" * 80)
        print("\nHalted training; reached {} training iterations.".format(timestep))
    except:
        print("-" * 80)
        print("\nUnknown error:")
        traceback.print_exc()
    finally:
        print("\nReached {0} training iterations out of max {1}.".format(timestep,max_iters))
        logger.close()
        train_dataset.close()
        #valid_dataset.close()
        print("Log file handles and dataset handles closed.")

if __name__ == '__main__':
    # read CLI args:
    parser = argparse.ArgumentParser(description="Train a new WaveNet Speech-to-Text model.")
    parser.add_argument("--config", dest="config", help="Path to JSON config file.")
    args = parser.parse_args()
    config = json_to_config(args.config)
    config_to_json(config, os.path.join(config['training']['save_dir'], "config.json"))
    train(config)
