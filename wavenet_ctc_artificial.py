"""
Train WaveNet-CTC to overfit on artificial signal-to-sequence data.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

from warpctc_pytorch import CTCLoss
from modules.wavenet import WaveNet
from modules.classifier import WaveNetClassifier
from utils.loaders import Loader


### Training parameters, etc.
num_iterations = 100000
num_core_epochs = 30
num_ctc_epochs = 50
wavenet_dils = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
classifier_layers = [(256, 256, 2, d) for d in [1, 2, 4, 8, 16, 32] ]
downsample_rate = 3
num_labels = 5 # == |{A,G,C,T,-}|
num_levels = 256
dataset_path = "./data/artificial.hdf5"
wavenet_model_save_path = "./runs/artificial/wavenet_model.pth"
classifier_model_save_path = "./runs/artificial/classifier_model.pth"
print_every = 10

### construct wavenet and classifier models:
wavenet = WaveNet(num_levels, 2,
                  [(num_levels, num_levels, 2, d) for d in wavenet_dils],
                  num_levels, softmax=False)
classifier = WaveNetClassifier(num_levels, num_labels, classifier_layers,
                               pool_kernel_size=downsample_rate,
                               input_kernel_size=2, input_dilation=1,
                               out_kernel_size=2, out_dilation=1, softmax=False)


### construct data loader:
dataloader = Loader(dataset_path, num_signal_levels=num_levels, max_iters=num_iterations,
                    num_epochs=(num_core_epochs+num_ctc_epochs))


### update to CUDA if available:
if torch.cuda.is_available():
    wavenet.cuda()
    classifier.cuda()
    dataloader.cuda()
    print("Placed WaveNet, Classifier Network, and DataLoader on CUDA.")


### construct loss functions:
ctc_loss_fn = CTCLoss()
xe_loss_fn = nn.CrossEntropyLoss()


### construct optimizers:
wavenet_lr = None
ctc_lr = None
# (... more settings here ...)
wavenet_optimizer = optim.RMSprop(wavenet.parameters())
ctc_optimizer = optim.Adadelta(classifier.parameters())


### (pre-)training closures:
def pretrain_core_wavenet(sig):
    """Train the wavenet against itself in parallel."""
    wavenet_optimizer.zero_grad()
    pred_sig = wavenet(sig[:,:,:-1])
    _, dense_sig = torch.max(sig[:,:,1:], dim=1)
    xe_loss = 0.
    for t in range(sig.size(2)-1):
        xe_loss = xe_loss + xe_loss_fn(pred_sig[:,:,t], dense_sig[:,t])
    xe_loss.backward()
    wavenet_optimizer.step()
    return xe_loss.data[0]

def train_ctc_network(sig, seq):
    """Train the wavenet & ctc-classifier jointly against both losses."""
    wavenet_optimizer.zero_grad()
    ctc_optimizer.zero_grad()
    pred_sig = wavenet(sig)
    transcription = classifier(pred_sig)
    #-- cross entropy loss on wavenet output:
    _, dense_sig = torch.max(sig[:,:,1:], dim=1)
    xe_loss = 0.
    for t in range(sig.size(2)-1):
        xe_loss = xe_loss + xe_loss_fn(pred_sig[:,:,t], dense_sig[:,t])
    #-- ctc loss on predicted transcriptions
    probs = transcription.permute(2,0,1).contiguous() # expects (sequence, batch, logits)
    prob_lengths = torch.IntTensor([probs.size(0)])
    labels = seq[0].cpu().int() # expects flattened labels
    label_lengths = torch.IntTensor([len(labels)])
    ctc_loss = ctc_loss_fn(probs, labels, prob_lengths, label_lengths)
    #-- backprop:
    total_loss = xe_loss + ctc_loss
    total_loss.backward()
    ctc_optimizer.step()
    wavenet_optimizer.step()
    return total_loss.data[0]


try:
    while True:
        if dataloader.epochs < num_core_epochs:
            signal, _ = dataloader.fetch(bucket=0)
            xe_loss_train = pretrain_core_wavenet(signal)
            if dataloader.counter % print_every == 0:
                print("WaveNet-Pretrain XE Loss @ step {0}: {1}".format(dataloader.counter, xe_loss_train))
        else:
            signal, sequence = dataloader.fetch(bucket=0)
            train_ctc_network(signal, sequence)
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
finally:
    # cleanup:
    print("Closing file handles...")
    dataloader.close()
    print("Saving models...")
    torch.save(wavenet.state_dict(), wavenet_model_save_path)
    torch.save(classifier.state_dict(), classifier_model_save_path)
    print("...Done.")
