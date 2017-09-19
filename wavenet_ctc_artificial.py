"""
Train WaveNet-CTC to overfit on artificial signal-to-sequence data.

* N.B.: regarding the downsample rate: this must be set low enough so that
the CTC layer still has enough timesteps to insert blank labels between
repeats.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

from warpctc_pytorch import CTCLoss
from modules.wavenet import WaveNet
from modules.classifier import WaveNetClassifier
from utils.loaders import PoreModelLoader

import traceback

### Training parameters, etc.
num_iterations = 300000
num_core_epochs = 0
num_ctc_epochs = 50
epoch_size = 3000
batch_size = 16
wavenet_dils = [1] * 12
classifier_dils = [1] * 12
downsample_rate = 1
num_labels = 5 # == |{-,A,G,C,T}|
out_dim = 256
num_levels = 256
dataset_path = "./data/artificial.large.hdf5"
wavenet_model_save_path = "./runs/artificial/wavenet_model.5.pth"
classifier_model_save_path = "./runs/artificial/classifier_model.5.pth"
wavenet_model_restore_path = "./runs/artificial/wavenet_model.3.pth"
classifier_model_restore_path = "./runs/artificial/classifier_model.3.pth"
restore_wavenet = False #True # if true, restore a model
restore_classifier = False #True # if true, restore a model
print_every = 10
num_early_stop_counts = 3
early_stop_threshold = 0.8

### construct wavenet and classifier models:
wavenet = WaveNet(num_levels, 2,
                  [(num_levels, num_levels, 2, d) for d in wavenet_dils],
                  num_levels, softmax=False)
classifier = WaveNetClassifier(num_levels, num_labels,
                               [(num_levels, num_levels, 3, d) for d in classifier_dils],
                               out_dim,
                               pool_kernel_size=downsample_rate,
                               input_kernel_size=2, input_dilation=1,
                               softmax=False)

### if restore models, load their saved values:
if restore_wavenet:
    wavenet.load_state_dict(torch.load(wavenet_model_restore_path))
    print("Restored WaveNet weights from: {}".format(wavenet_model_restore_path))
if restore_classifier:
    classifier.load_state_dict(torch.load(classifier_model_restore_path))
    print("Restored classifier weights from: {}".format(classifier_model_restore_path))

### construct data loader:
dataloader = PoreModelLoader(num_iterations, (num_core_epochs+num_ctc_epochs), epoch_size, batch_size,
                             lengths=(90,100), sample_noise=10.)


### update to CUDA if available:
CUDA_FLAG = False
if torch.cuda.is_available():
    wavenet.cuda()
    classifier.cuda()
    CUDA_FLAG = True
    print("Placed WaveNet & CTC Network on CUDA.")


### construct loss functions:
ctc_loss_fn = CTCLoss()
xe_loss_fn = nn.CrossEntropyLoss()


### construct optimizers:
joint_optimizer = optim.Adagrad([{'params': wavenet.parameters()},
                                 {'params': classifier.parameters()}],
                                lr=0.000005)


### (pre-)training closures:
def pretrain_core_wavenet(sig):
    """Train the wavenet against itself in parallel."""
    joint_optimizer.zero_grad()
    pred_sig = wavenet(sig[:,:,:-1])
    _, dense_sig = torch.max(sig[:,:,1:], dim=1)
    xe_loss = 0.
    for t in range(sig.size(2)-1):
        xe_loss = xe_loss + xe_loss_fn(pred_sig[:,:,t], dense_sig[:,t])
    avg_xe_loss = xe_loss / sig.size(2)
    avg_xe_loss.backward()
    joint_optimizer.step()
    return xe_loss.data[0]

def train_ctc_network(sig, seq, seq_lengths):
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
    labels = labels + Variable(torch.ones(seq[0].size()).int())
    ctc_loss = ctc_loss_fn(probs, labels, prob_lengths, seq_lengths)
    #-- backprop - choose which loss to optimize: total sum, average sum, ctc
    total_loss = xe_loss + ctc_loss
    average_loss = (xe_loss / sig.size(2)) + (ctc_loss / transcription.size(2))
    average_ctc_loss = (ctc_loss / transcription.size(2))
    #total_loss.backward()
    average_loss.backward()
    #average_ctc_loss.backward()
    #ctc_loss.backward()
    #-- apply gradients and return loss values (for output logs in training loop):
    joint_optimizer.step()
    return (total_loss.data[0], xe_loss.data[0], ctc_loss.data[0])


### main training loop:
try:
    while True:
        if dataloader.epochs < num_core_epochs:
            signal, _, _ = dataloader.fetch()
            if CUDA_FLAG: signal = signal.cuda()
            xe_loss_train = pretrain_core_wavenet(signal)
            if dataloader.counter % print_every == 0:
                loss_per_sample = xe_loss_train / int(signal.size(2))
                print("Step: {0:5d} | Pretrain XE Loss Tot: {1:07.4f} | Per-Sample: {2:07.4f}".format(
                    dataloader.counter, xe_loss_train, loss_per_sample))
        else:
            signal, sequence, lengths = dataloader.fetch()
            if CUDA_FLAG: signal = signal.cuda()
            l_total, l_xe, l_ctc = train_ctc_network(signal, sequence, lengths)
            if dataloader.counter % print_every == 0:
                total_pc = l_total / int(sequence.size(0))
                xe_pc = l_xe / int(signal.size(2))
                ctc_pc = l_ctc / int(sequence.size(0))
                print(("Step: {0:5d} | XE+CTC Loss Tot: {1:07.4f}={2:07.4f}+{3:07.4f} | " + \
                       "Per-Sample XE: {4:07.4f} | Per-Char CTC: {5:07.4f}").format(
                           dataloader.counter, l_total, l_xe, l_ctc, xe_pc, ctc_pc))
                # implement early stopping:
                if ctc_pc < early_stop_threshold:
                    num_early_stop_counts += 1
                    if num_early_stop_counts == 3:
                        print("Early stopping!")
                        raise StopIteration
                else:
                    num_eary_stop_counts = 0
except StopIteration:
    # handle stopiteration from dataloader (finished all iterations)
    print("Completed all epochs/iterations.")
    print("Num epochs: {}".format(dataloader.epochs))
    print("Num iterations: {}".format(dataloader.counter))
except KeyboardInterrupt:
    # handle manual halt from keyboard
    print("Training manually interrupted from keyboard.")
    print("Num epochs: {}".format(dataloader.epochs))
    print("Num iterations: {}".format(dataloader.counter))
except Exception as e:
    # handle all other exceptions:
    print("Something went wrong; received the following error:")
    print(e)
    traceback.print_exc()
finally:
    # cleanup:
    print("Saving models...")
    torch.save(wavenet.state_dict(), wavenet_model_save_path)
    torch.save(classifier.state_dict(), classifier_model_save_path)
    print("...Done.")
