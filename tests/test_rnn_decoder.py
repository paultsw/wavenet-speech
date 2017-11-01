"""
Tests for RNN decoder and StackedLSTMCell.
"""
import torch
from torch.autograd import Variable
import torch.optim as optim
from modules.rnn_decoder import StackedLSTMCell, RNNByteNetDecoder

###___________________________________________________________________
# Stacked LSTM Cell Tests
batch_size = 8
hidden_dim = 12
num_layers = 5

lstm_cell = StackedLSTMCell(hidden_dim, num_layers)

num_timesteps = 100
incr_range = torch.arange(1, num_timesteps) # TODO: [num_timesteps x batch_size x hidden_dim]
source_seq = Variable(incr_range.mul(0.5))
target_seq = Variable(incr_range)

x0 = Variable(torch.randn(batch_size,hidden_dim).mul(0.0001))
h0s = [Variable(torch.randn(batch_size,hidden_dim)) for _ in range(num_layers)]
c0s = [Variable(torch.randn(batch_size,hidden_dim)) for _ in range(num_layers)]

opt = optim.Rmsprop(lstm_cell.parameters())
loss_fn = nn.MSELoss()

# loop over timesteps, predict target sequence:
num_train_steps = 1000
print_every = 10
for k in range(num_train_steps):
    opt.zero_grad()
    preds = []
    loss = 0.0
    for t in range(num_timesteps):
        xt = x0 if (t == 0) else x
        hts = h0s if (t == 0) else hs
        cts = c0s if (t == 0) else cs
        x, hs, cs = lstm_cell(xt,hts,cts)
        loss = loss + loss_fn(x,target_seq[t,:])
    loss.backward()
    if (k % print_every == 0): print("Step: {0} | Loss: {1}".format(k,loss.data[0])
    opt.step()


###___________________________________________________________________
# RNN ByteNet-Style Decoder Tests
dec = RNNByteNetDecoder(7, 24, 32, 64, 5, max_timesteps=100)
x0 = Variable(torch.randn(8).clamp(min=0.,max=1.).mul(7).long())
enc_seq = Variable(torch.randn(8,24,50))
hvals = [Variable(torch.randn(8,32)) for _ in range(5)]
cvals = [Variable(torch.randn(8,32)) for _ in range(5)]
