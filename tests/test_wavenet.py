"""
Tests for WaveNet module.
"""
import torch
from torch.autograd import Variable

from modules.wavenet import WaveNet

### parameters
kwidth = 2 # constant kernel width across all convolutional layers
batch_size = 5
seq_length = 14
seq_dim = 11 # for both input and output, since we're feeding one-hot and outputting softmax distributions
#dilations = [1,2]
dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
layers = [(seq_dim, seq_dim, kwidth, d) for d in dilations]

### input sequence:
input_seq = Variable(torch.randn(batch_size, seq_dim, seq_length))

### construct WaveNet:
# (turn softmax off because torch.nn.CrossEntropyLoss does it for us)
wavenet = WaveNet(seq_dim, kwidth, layers, seq_dim, softmax=False)

### run wavenet on inputs:
predictions = wavenet(input_seq)

### prints, asserts, etc:
print(predictions.size())
print(predictions)
assert (torch.Size([batch_size, seq_dim, seq_length]) == predictions.size())

# run overfitting test on constant sequence:
target_sequence = torch.rand(batch_size, seq_length).mul_(seq_dim).long()
source_sequence = torch.zeros(batch_size, seq_dim, seq_length).scatter_(1, target_sequence.unsqueeze(1), 1)
source = Variable(source_sequence)
target = Variable(target_sequence)
print("===== Attempting to overfit on constant sequences... =====")
num_iterations = 1000
print_every = 50
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(wavenet.parameters())
try:
    for step in range(num_iterations):
        optimizer.zero_grad()
        preds = wavenet(source)
        loss = 0.
        for t in range(target.size(1)):
            loss = loss + loss_fn(preds[:,:,t], target[:,t])
        if step % print_every == 0: print("Loss @ step {0}: {1}".format(step,loss.data[0]))
        loss.backward()
        optimizer.step()
except KeyboardInterrupt:
    print("...Halted training.")
finally:
    print("... Done. You should see a gradually decreasing loss. (If you don't, it means that something went wrong.)")
