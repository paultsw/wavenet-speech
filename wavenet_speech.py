"""
Main entrypoint for wavenet.
"""
# import abstract classes that unify different model/data types:
from Model import Model
from Optimizer import Optimizer
from Loss import Loss
from Decoder import Decoder
from Dataset import Dataset

# import utilities:
import argparse
from utils.config_tools import json_to_config, config_to_json
from utils.logging import Logger

# autodetect CUDA availability:
_CUDA_ = torch.cuda.is_available()


def evaluate(cfg):
    """
    Evaluate a pre-trained model on a dataset.
    """
    # construct necessary components for training:
    model = Model(cfg['model'])
    loss_fn = Loss(cfg['loss_choice'], averaged=True)
    dataset = Dataset(cfg['datatype'], dataset=cfg['dataset_path'])
    decoder = Decoder(cfg['decode_type'], cfg['batch_size'], cfg['num_labels'], beam_width=cfg['beam_width'])
    for t in range(cfg['num_iterations']):
        sigs, seqs, lengths = dataset.fetch()
        _, transcriptions = model.predict(sigs)
        _, loss = loss_fn.calculate(None, None, transcriptions, seqs, lengths, avg=True)
        probas, decoded = decoder.decode(transcriptions)
        print("Loss @ iteration {0}: {1}".format(t,loss.data[0]))
        print("Decoded hypotheses:")
        for hyp in decoded: print(hyp)
        print("Probabilities:")
        print(probas)


def train(cfg):
    """
    Train a new model.
    """
    # construct necessary components for training:
    model = Model(cfg['model'])
    loss_fn = Loss(cfg['loss_choice'], averaged=True)
    dataset = Dataset(cfg['datatype'], dataset=cfg['dataset_path'])
    opt = Optimizer(model.get_parameters(), cfg['optim_type'], cfg['lr'])
    logger = Logger(cfg['run_dir'])

    # see if we want to restore a model:
    if not (cfg['restore_model_base'] is None) and not (cfg['restore_model_ctc'] is None):
        model.restore(cfg['restore_model_base'], cfg['restore_model_ctc'])

    # training loop:
    try:
        running_vlosses = [10000]
        for k in range(cfg['num_epochs']):
            #-- train for one epoch:
            for t in range(cfg['epoch_size']):
                opt.zero_grad()
                sigs, seqs, lengths = dataset.fetch(train_or_valid='train')
                pred0, pred1 = model.predict(sigs)
                loss = loss_fn.calculate(sigs, pred0, pred1, seqs, lengths, avg=True)
                loss[1].backward()
                opt.step()
                if t % cfg['print_every'] == 0: print("Step {0} Training Loss: {1}".format(t,loss[1].data[0]))
            #-- validate:
            validation_losses = []
            for t in range(cfg['num_validation_steps']):
                sigs, seqs, lengths = dataset.fetch(train_or_valid='valid')
                pred0, pred1 = model.predict(sigs)
                loss = loss_fn.calculate(sigs, pred0, pred1, seqs, lengths, avg=True)
                validation_losses.append(loss[1].data[0])
                if t % cfg['print_every'] == 0: print("Step {0} Validation Loss: {1}".format(t,loss[1].data[0]))
            vloss = (sum(validation_losses) / cfg['num_validation_steps'])
            #-- post-epoch decisions: decide whether or not to lower LR, early-stopping, etc:
            if vloss < cfg['early_stopping_loss_threshold']: raise StopIteration
            if (running_vlosses[-1] - vloss) < cfg['reduce_lr_threshold']: opt.reduce_lr()

    # handle StopIteration:
    except StopIteration:
        print("Finished all data iterations and/or early-stopping detected.")

    # handle manual interrupt from keyboard:
    except KeyboardInterrupt:
        print("Training halted from keyboard.")

    # handle generic exceptions:
    except:
        print("Unrecognized error detected.")
        traceback.print_exc()

    # cleanup:
    finally:
        print("Number of epochs: {0}; Number of iterations: {1}".format(dataset.epochs, dataset.ctr))
        print("Saving models to disk and closing dataset...")
        model.save(cfg['save_model_base'], cfg['save_model_ctc'])
        dataset.close()
        print("... All done.")



def bayesopt(**kwargs):
    """
    A specialized function for Bayesian optimization of models.
    
    (Import this function instead of the other two when using BayesOpt for
    model selection.)
    """
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument("task", choices=('evaluate', 'train'), help="TODO: help string")
    parser.add_argument("config", help="Path to JSON file containing training/evaluation config.")
    args = parser.parse_args()
    if args.task == 'evaluate': evaluate(json_to_config(args.config))
    if args.task == 'train': train(json_to_config(args.config))
