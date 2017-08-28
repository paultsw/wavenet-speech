"""
Test CausalConv1d module.
"""
import torch
from torch.autograd import Variable

from modules.conv_ops import CausalConv1d, compute_new_length

# param settings:
in_channels = 3
out_channels = 3
kernel_width = 5
seq_length = 15
dilation_rate = 3
batch_size = 2

# example input seq:
in_seq_ = torch.randn(batch_size, in_channels, seq_length)
in_seq = Variable(in_seq_)

# run causal conv1d:
conv = CausalConv1d(in_channels, out_channels, kernel_width, dilation=dilation_rate)
out_seq = conv(in_seq)

# compute expected length before the conv1d:
padding = (kernel_width-1)*dilation_rate
expected_length = int(compute_new_length(seq_length, padding, dilation_rate, kernel_width) - padding)

# assert that output is same length:
print(out_seq.size())
print(torch.Size([batch_size, out_channels, seq_length]))
print(torch.Size([batch_size, out_channels, expected_length]))
assert (torch.Size([batch_size, out_channels, seq_length]) == out_seq.size())
assert (torch.Size([batch_size, out_channels, expected_length]) == out_seq.size())

# run overfitting test on constant sequence:
source_sequence = Variable(torch.ones(batch_size, in_channels, seq_length).mul_(2.))
target_sequence = Variable(torch.ones(batch_size, out_channels, seq_length).mul_(3.))
print("===== Attempting to overfit on constant sequences... =====")
num_iterations = 1000
print_every = 50
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(conv.parameters())
for step in range(num_iterations):
    optimizer.zero_grad()
    preds = conv(source_sequence)
    loss = loss_fn(preds, target_sequence)
    if step % print_every == 0: print("Loss @ step {0}: {1}".format(step,loss.data[0]))
    loss.backward()
    optimizer.step()
print("... Done. You should see a gradually decreasing loss.")
