"""
Tests for Residual Block.
"""
import torch
from torch.autograd import Variable

from modules.block import ResidualBlock

### parameters:
in_channels = 4
out_channels = 5
sequence_length = 12
kernel_width = 2
dilation = 2
batch_size = 3

### construct input sequence:
in_seq = Variable(torch.randn(batch_size, in_channels, sequence_length))

### construct residual block and fetch outputs:
residual_block = ResidualBlock(in_channels, out_channels, kernel_width, dilation)
out_seq, skip_connection = residual_block(in_seq)

noncausal_block = ResidualBlock(in_channels, out_channels, kernel_width, dilation, causal=False)
nc_out_seq, nc_skip_connection = noncausal_block(in_seq)

### assertions and prints:
print("===== Assert that causal dimensions are correct... =====")
print(out_seq.size())
print(skip_connection.size())

assert (torch.Size([batch_size, out_channels, sequence_length]) == out_seq.size())
assert (torch.Size([batch_size, out_channels, sequence_length]) == skip_connection.size())

print("===== Assert that non-causal dimensions are correct... =====")
print(nc_out_seq.size())
print(nc_skip_connection.size())

assert (torch.Size([batch_size, out_channels, sequence_length]) == nc_out_seq.size())
assert (torch.Size([batch_size, out_channels, sequence_length]) == nc_skip_connection.size())

# run overfitting test on constant sequence:
print("===== Attempting to overfit on constant sequences with causal block... =====")
source_sequence = Variable(torch.ones(batch_size, in_channels, sequence_length).mul_(2.))
target_sequence = Variable(torch.ones(batch_size, out_channels, sequence_length).mul_(3.))
num_iterations = 1000
print_every = 50
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(residual_block.parameters())
for step in range(num_iterations):
    optimizer.zero_grad()
    outs, skips = residual_block(source_sequence)
    loss = loss_fn(outs+skips, target_sequence)
    if step % print_every == 0: print("Loss @ step {0}: {1}".format(step,loss.data[0]))
    loss.backward()
    optimizer.step()
print("... Done. You should see a gradually decreasing loss. (If you don't, something went wrong.)")

print("===== Attempting to overfit on constant sequences with non-causal block... =====")
source_sequence = Variable(torch.ones(batch_size, in_channels, sequence_length).mul_(2.))
target_sequence = Variable(torch.ones(batch_size, out_channels, sequence_length).mul_(3.))
num_iterations = 1000
print_every = 50
loss_fn = torch.nn.MSELoss()
nc_optimizer = torch.optim.Adam(noncausal_block.parameters())
for step in range(num_iterations):
    nc_optimizer.zero_grad()
    outs, skips = noncausal_block(source_sequence)
    loss = loss_fn(outs+skips, target_sequence)
    if step % print_every == 0: print("Loss @ step {0}: {1}".format(step,loss.data[0]))
    loss.backward()
    nc_optimizer.step()
print("... Done. You should see a gradually decreasing loss. (If you don't, something went wrong.)")

#___________________________________________________________________________________________________

##### Run CUDA tests if available:
import sys
if not (torch.cuda.is_available()): sys.exit()
print("GPU with CUDA support detected. Running CUDA tests...")

### construct input sequence:
in_seq = Variable(torch.randn(batch_size, in_channels, sequence_length).cuda())

### construct residual block and fetch outputs:
residual_block = ResidualBlock(in_channels, out_channels, kernel_width, dilation).cuda()
out_seq, skip_connection = residual_block(in_seq)

noncausal_block = ResidualBlock(in_channels, out_channels, kernel_width, dilation, causal=False).cuda()
nc_out_seq, nc_skip_connection = noncausal_block(in_seq)

### assertions and prints:
print("===== Assert that causal dimensions are correct... =====")
print(out_seq.size())
print(skip_connection.size())

assert (torch.Size([batch_size, out_channels, sequence_length]) == out_seq.size())
assert (torch.Size([batch_size, out_channels, sequence_length]) == skip_connection.size())

print("===== Assert that non-causal dimensions are correct... =====")
print(nc_out_seq.size())
print(nc_skip_connection.size())

assert (torch.Size([batch_size, out_channels, sequence_length]) == nc_out_seq.size())
assert (torch.Size([batch_size, out_channels, sequence_length]) == nc_skip_connection.size())

# run overfitting test on constant sequence:
print("===== Attempting to overfit on constant sequences with causal block... =====")
source_sequence = Variable(torch.ones(batch_size, in_channels, sequence_length).mul_(2.).cuda())
target_sequence = Variable(torch.ones(batch_size, out_channels, sequence_length).mul_(3.).cuda())
num_iterations = 1000
print_every = 50
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(residual_block.parameters())
for step in range(num_iterations):
    optimizer.zero_grad()
    outs, skips = residual_block(source_sequence)
    loss = loss_fn(outs+skips, target_sequence)
    if step % print_every == 0: print("Loss @ step {0}: {1}".format(step,loss.data[0]))
    loss.backward()
    optimizer.step()
print("... Done. You should see a gradually decreasing loss. (If you don't, something went wrong.)")

print("===== Attempting to overfit on constant sequences with non-causal block... =====")
source_sequence = Variable(torch.ones(batch_size, in_channels, sequence_length).mul_(2.).cuda())
target_sequence = Variable(torch.ones(batch_size, out_channels, sequence_length).mul_(3.).cuda())
num_iterations = 1000
print_every = 50
loss_fn = torch.nn.MSELoss()
nc_optimizer = torch.optim.Adam(noncausal_block.parameters())
for step in range(num_iterations):
    nc_optimizer.zero_grad()
    outs, skips = noncausal_block(source_sequence)
    loss = loss_fn(outs+skips, target_sequence)
    if step % print_every == 0: print("Loss @ step {0}: {1}".format(step,loss.data[0]))
    loss.backward()
    nc_optimizer.step()
print("... Done. You should see a gradually decreasing loss. (If you don't, something went wrong.)")
