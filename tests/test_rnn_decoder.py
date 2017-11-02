"""
Tests for RNN decoder and StackedLSTMCell.
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from modules.rnn_decoder import StackedLSTMCell, RNNByteNetDecoder

###___________________________________________________________________
# Stacked LSTM Cell Tests
print("========== ========== STACKED LSTM CELL TESTS ========== ==========")

batch_size = 8
hidden_dim = 12
num_layers = 2

lstm_cell = StackedLSTMCell(hidden_dim, num_layers)

num_timesteps = 50
const_seq = torch.zeros(num_timesteps,batch_size,hidden_dim).add(4.)
source_seq = Variable(const_seq.mul(0.5))
target_seq = Variable(const_seq)

h0s = [Variable(torch.ones(batch_size,hidden_dim)) for _ in range(num_layers)]
c0s = [Variable(torch.ones(batch_size,hidden_dim)) for _ in range(num_layers)]

opt = optim.RMSprop(lstm_cell.parameters(), lr=0.01)
loss_fn = nn.L1Loss()

# loop over timesteps, predict target sequence:
print("Attempting to predict a constant sequence...")
num_train_steps = 100
print_every = 10
for k in range(num_train_steps):
    opt.zero_grad()
    preds = []
    loss = 0.0
    for t in range(num_timesteps):
        hts = h0s if (t == 0) else hs
        cts = c0s if (t == 0) else cs
        o, hs, cs = lstm_cell(source_seq[t],hts,cts)
        loss = loss + loss_fn(o,target_seq[t,:,:])
    avg_loss = loss.div(num_timesteps)
    avg_loss.backward()
    if (k % print_every == 0): print("Step: {0} | Avg. Loss: {1}".format(k,avg_loss.data[0]))
    opt.step()
print("Done. (If you don't see a decreasing loss, something went wrong.)")

###___________________________________________________________________
# RNN ByteNet-Style Decoder Tests
print("========== ========== RNN-BYTENET-DECODER TESTS ========== ==========")

# set params:
bsz = 8
num_labels = 7
encoding_dim = 12
hidden_dim = 24
out_dim = 32
num_layers = 5
enc_length = 50

# construct decoder, loss, optimizer:
dec = RNNByteNetDecoder(num_labels, encoding_dim, hidden_dim, out_dim, num_layers, max_timesteps=100)
loss_fn = nn.CrossEntropyLoss()
opt = optim.RMSprop(dec.parameters())

# test single forward step:
x0 = Variable(torch.randn(bsz).clamp(min=0.,max=1.).mul(num_labels-1).long())
enc_seq = Variable(torch.randn(bsz, encoding_dim, enc_length).mul(0.001)) # small-magnitude random-normal seq
hvals = [Variable(torch.randn(bsz, hidden_dim)) for _ in range(num_layers)]
cvals = [Variable(torch.randn(bsz,hidden_dim)) for _ in range(num_layers)]
out, houts, couts = dec(x0, hvals, cvals, enc_step=enc_seq[:,:,0])
print(out)

# try to predict a simple target sequence:
print("Trying to learn to replicate a basic sequence...")
num_train_steps = 100
print_every = 5
target_seq = Variable(torch.zeros(bsz,enc_length).add(4).long()) # (constant seq of 4's)
for k in range(num_train_steps):
    opt.zero_grad()
    out_seq, lengths = dec.unfold(enc_seq)
    loss = 0.0
    for t in range(min(out_seq.size(0), target_seq.size(1))):
        loss = loss + loss_fn(out_seq[t], target_seq[:,t])
    loss.backward()
    if (k % print_every == 0): print("Step: {0} | Loss: {1}".format(k,loss.data[0]))
    opt.step()
print("...Done. (If you don't see a gradually decreasing loss, something went wrong.)")
