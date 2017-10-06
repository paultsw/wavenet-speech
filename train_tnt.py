"""
Train a model with a TorchNet engine.

Usage:
$ python -m visdom.server & python train_tnt.py
"""
# general utils:
from tqdm import tqdm
import argparse
from utils.config_tools import json_to_config, config_to_json
from os.path import exists
# torch utils:
import torch.nn as nn
from torch.autograd import Variable
# abstraction classes:
from Model import Model
from Dataset import Dataset
from Decoder import Decoder
from Optimizer import Optimizer
from Loss import Loss
# torchnet:
import torchnet as tnt
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger

##### Main training loop with loggers:
def main(cfg):
    #--- construct dataset from config and define data iterator function:
    dataset = Dataset(cfg['data_type'], dataset=cfg['dataset'])
    def get_iterator(train_mode=True):
        """Return a parallelized TensorDataset."""
        tds = tnt.dataset.Dataset(None) # [TODO: figure out what kind of tnt.Dataset should go here]
        return tds.parallel(batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=train_mode)

    #--- construct model from config:
    # `Model` exposes `Model.{predict(), get_parameters(), restore(), save()}`
    model = Model(cfg['model_type'], cfg['model_cfg'])

    #--- construct optim:
    # `Optimizer` exposes `Optimizer.{adjust_lr(), step(), zero_grad()}`
    optimizer = Optimizer(model.get_parameters(), cfg['optim_type'], cfg['lr'], weight_decay=None, reduce_lr=None)

    #--- construct beam search decoder:
    # `Decoder` exposes `Decoder.decode()`, which expects `logits ~ (batch, nlabels, seq)`,
    # as well as `Decoder.refresh_beams()`.
    decoder = Decoder(cfg['decoder_type'], cfg['batch_size'], cfg['num_labels'],
                      beam_width=cfg['beam_width'], cuda=False)

    #--- construct engine/meters/loggers:
    engine = Engine()
    # capture running loss' average value and stdv:
    loss_meter = tnt.meter.AverageValueMeter()
    # train/validation loss plots:
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    valid_loss_logger = VisdomPlotLogger('line', opts={'title': 'Validation Loss'})
    # log text for decoded outputs after each epoch @ validation time:
    valid_decode_logger = VisdomTextLogger(update_type='APPEND')

    #--- define loss function and loss compute:
    # `Loss` exposes `Loss.calculate()`
    loss = Loss(cfg['loss_choice'], ce_weights=None, joint_balance=None, averaged=True)
    def run_model(sample):
        input_seqs, target_seqs, target_lengths = sample
        outs = model.predict(input_seqs)
        return loss.calculate(None, None, outs, target_seqs, target_lengths), outs

    #--- reset meters:
    def reset_all_meters():
        """Reset all meters."""
        loss_meter.reset()

    #--- sample hook:
    def on_sample(state):
        """What to do after each sample (after obtaining a data sample)."""
        state['sample'].append(state['train'])

    #--- on update:
    def on_update(state):
        """What to do after each SGD update. [Training only.]"""
        pass # [Don't do anything for now.]

    #--- forward pass hook:
    def on_forward(state):
        """Update loggers at each forward pass. [Testing only.]"""
        meter_loss.add(state['loss'].data[0])
    
    #--- start epoch hook:
    def on_start_epoch(state):
        """At the start of each epoch. [Training only.]"""
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])
    
    #--- end epoch hook:
    def on_end_epoch(state):
        """After each epoch, perform validation and do a beam search decoding printout. [Training only.]"""
        # Log training info to loggers:
        # [TODO: log info to training loggers here]

        # run validation:
        reset_meters()
        engine.test(run_model, get_iterator(train_mode=False))
        # [TODO: log info to validation loggers here]
        # [TODO: run beam search decoding here]

    #--- set up train engine and start training:
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(run_model, get_iterator(train_mode=True), max_epochs=cfg['max_epochs'], optimizer=optimizer)


if __name__ == '__main__':
    # Ingest config file:
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--config", dest="config", help="Path to configuration JSON.")
    args = parser.parse_args()
    cfg = json_to_config(args.config)
    config_to_json(cfg, cfg['run_dir'])
    main(cfg)
