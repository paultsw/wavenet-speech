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
from utils.loader import OverfitLoader # [TODO: swap this with Loader class once completed]
from utils.logging import Logger
from utils.config_tools import json_to_config, config_to_json

# argument parsers:
import argparse
import os
import traceback

def wavenet_train_step(wave, classifier, nll_fn, ctc_fn, sig, seq, wave_opt, class_opt):
    """Take one training step of WaveNet."""
    # 0. clear gradients:
    wave_opt.zero_grad()
    class_opt.zero_grad()
    
    # 1. run wavenet on signal:
    wavenet_pred = wave(sig[:,:,0:-1])

    # 2. run classifier on wavenet output distribution:
    classifier_pred = classifier(wavenet_pred)

    # 3. compute NLL between true signal and wavenet output:
    wavenet_loss = nll_fn(wavenet_pred, sig[:,:,1:])

    # 4. compute CTC loss between true seq and predicted seq:
    classifier_loss = ctc_fn(classifier_pred, seq)
    
    # 5. backprop:
    joint_loss = wavenet_loss + classifier_loss
    joint_loss.backward()
    class_opt.step()
    wave_opt.step()

    # 6. return loss values:
    return (wavenet_loss, classifier_loss, joint_loss)


def build_optimizer(optim_cfg, params):
    """Build an optimizer based on config settings."""
    optim_type = optim_cfg['type']
    lr = optim_cfg['learning_rate']
    wd = optim_cfg['wd']
    max_grad_norm = optim_cfg['max_grad_norm']
    assert (optim_type in ['adam', 'rmsprop'])
    if optim_type == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=wd)
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
    wavenet_layers = config['model']['base']['layers']
    # classifier settings:
    num_labels = config['model']['classifier']['num_labels']
    classifier_layers = config['model']['classifier']['layers']
    downsample_rate = config['model']['classifier']['downsample']
    wavenet_core = WaveNet(signal_dim, entry_kwidth, wavenet_layers, signal_dim, softmax=True)
    wavenet_class = WaveNetClassifier(signal_dim, num_labels, classifier_layers, pool_kernel_size=downsample)

    ###===== optionally restore weights to continue training:
    restore_wavenet_path = config['training']['restore_wavenet_path']
    restore_classifier_path = config['training']['restore_classifier_path']
    if restore_wavenet_path is not None:
        wavenet_core.load_state_dict(torch.load_path(restore_wavenet_path, map_location=lambda storage, loc: storage))
    if restore_classifier_path is not None:
        wavenet_class.load_state_dict(torch.load_path(restore_classifier_path, map_location=lambda storage, loc: storage))

    ###===== construct loss functions:
    nll_loss_fn = nn.NLLLoss()
    ctc_loss_fn = CTCLoss()

    ###===== construct optimizers:
    optim_base_cfg = config['training']['optim']['base']
    optim_classifier_cfg = config['training']['optim']['classifier']
    waveopt = build_optimizer(optim_base_cfg, wavenet_class.parameters())
    classopt = build_optimizer(optim_classifier_cfg, wavenet_core.parameters())

    ###===== construct data loader and logging object:
    train_dataset_path = config['training']['training_data']
    validation_dataset_path = config['training']['validation_data']
    save_dir = config['training']['save_dir']
    # [TODO: swap this with a real loader once completed]
    dataset = OverfitLoader("./data/overfit/signal.npy", "./data/overfit/read.npy")
    logger = Logger(save_dir)

    ###===== run training loop:
    print_every = config['training']['print_every']
    save_every = config['training']['save_every']
    num_epochs = config['training']['num_epochs']
    max_iters = config['training']['max_iters']
    timestep = 0
    try:
        for signal, sequence in dataset:
            nll_loss, ctc_loss, joint_loss = wavenet_train_step(
                wavenet_core, wavenet_class, nll_loss_fn, ctc_loss_fn, signal, sequence, waveopt, classopt)
            if timestep % print_every == 0:
                valid_signal, valid_sequence = dataset.fetch_validation_data()
                valid_nll_loss, valid_ctc_loss, valid_joint_loss = wavenet_eval_step(
                    wavenet_core, wavenet_class, nll_loss_fn, ctc_loss_fn, valid_signal, valid_sequence)
                logger.log("NLL", timestep, nll_loss, valid_nll_loss)
                logger.log("CTC", timestep, ctc_loss, valid_ctc_loss)
                logger.log("TOT", timestep, joint_loss, joint_ctc_loss)
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
        dataset.close()
        print("Log file handles and dataset handles closed.")

if __name__ == '__main__':
    # read CLI args:
    parser = argparse.ArgumentParser(description="Train a new WaveNet Speech-to-Text model.")
    parser.add_argument("--config", dest="config", help="Path to JSON config file.")
    args = parser.parse_args()
    config = json_to_cfg(args.config)
    config_to_json(config, os.path.join(config['training']['save_dir'], "config.json"))
    train(config)
