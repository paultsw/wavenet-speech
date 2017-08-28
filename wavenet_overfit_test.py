"""
Overfit the core wavenet model on a single signal tensor.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from modules.wavenet import WaveNet

num_iterations = 100000

# load signal from data and create dataset:
signal = torch.load("./data/overfit/one_hot_signal.pth")
_, dense_signal = torch.max(signal[:,:,1:], dim=1)
source_seq = Variable(signal)
target_seq = Variable(dense_signal)

# construct model and optimizer:
dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
wavenet = WaveNet(256, 2, [(256, 256, 2, d) for d in dilations], 256, softmax=True)
learning_rate = 0.01
wd = 0.0001
betas = (0.9, 0.999)
optimizer = optim.Adam(wavenet.parameters(), lr=learning_rate, betas=betas, eps=1e-08, weight_decay=wd)
loss_fn = nn.CrossEntropyLoss()

# run training loop and occasionally print losses:
print_every = 100
best_observed_loss = 10**10
try:
    for step in range(num_iterations):
        optimizer.zero_grad()
        pred_dist_seq = wavenet(source_seq)
        loss = 0.
        for t in range(target_seq.size(1)):
            loss = loss + loss_fn(pred_dist_seq[:,:,t], target_seq[:,t])
        loss.backward()
        optimizer.step()
        if (step % print_every == 0):
            scalar_loss = loss.data[0]
            print("Step: {}".format(step))
            print("Total loss on this iteration: {}".format(scalar_loss))
            print("Average loss per sample on this iteration: {}".format(scalar_loss / target_seq.size(1)))
            if best_observed_loss > scalar_loss: best_observed_loss = scalar_loss
except KeyboardInterrupt:
    print("....Training interrupted.")
finally:
    torch.save(wavenet.state_dict(), "./data/overfit/wavenet_overfit_model.pth")
    print("Best observed loss (sampled): {}".format(best_observed_loss))
    
