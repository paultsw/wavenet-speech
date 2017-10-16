"""
Tests for ResidualMUBlock & ResidualReLUBlock.
"""
import torch
from torch.autograd import Variable
from modules.block import ResidualMUBlock, ResidualReLUBlock

### parameters:
nchannels = 4 # (this value must be even for these residual blocks)
sequence_length = 12
kernel_width = 2
dilation = 2
batch_size = 3

### construct input sequence:
in_seq = Variable(torch.randn(batch_size, nchannels, sequence_length))

### construct residual blocks and fetch outputs:
relu_block = ResidualReLUBlock(nchannels, kernel_width, dilation)
relu_block.init() # (initialize parameters)
relu_out_seq = relu_block(in_seq)

mu_block = ResidualMUBlock(nchannels, kernel_width, dilation)
mu_block.init() # (initialize parameters)
mu_out_seq = mu_block(in_seq)

### assertions and prints:
print("===== Assert that ReLU block output dimensions are correct... =====")
print(relu_out_seq.size())
assert (relu_out_seq.size() == in_seq.size())
print("===== Assert that MU block output dimensions are correct... =====")
print(mu_out_seq.size())
assert (mu_out_seq.size() == in_seq.size())

# run overfitting test on constant sequence:
print("===== Attempting to overfit on constant sequences with ReLU block... =====")
source_sequence = Variable(torch.ones(batch_size, nchannels, sequence_length).mul_(2.))
target_sequence = Variable(torch.ones(batch_size, nchannels, sequence_length).mul_(3.))
num_iterations = 1000
print_every = 50
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(relu_block.parameters())
for step in range(num_iterations):
    optimizer.zero_grad()
    outseq = relu_block(source_sequence)
    loss = loss_fn(outseq, target_sequence)
    if step % print_every == 0: print("Loss @ step {0}: {1}".format(step,loss.data[0]))
    loss.backward()
    optimizer.step()
print("... Done. You should see a gradually decreasing loss. (If you don't, something went wrong.)")

print("===== Attempting to overfit on constant sequences with MU block... =====")
source_sequence = Variable(torch.ones(batch_size, nchannels, sequence_length).mul_(2.))
target_sequence = Variable(torch.ones(batch_size, nchannels, sequence_length).mul_(3.))
num_iterations = 1000
print_every = 50
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mu_block.parameters())
for step in range(num_iterations):
    optimizer.zero_grad()
    outseq = mu_block(source_sequence)
    loss = loss_fn(outseq, target_sequence)
    if step % print_every == 0: print("Loss @ step {0}: {1}".format(step,loss.data[0]))
    loss.backward()
    optimizer.step()
print("... Done. You should see a gradually decreasing loss. (If you don't, something went wrong.)")

#___________________________________________________________________________________________________
##### Run CUDA tests if available:
import sys
if not (torch.cuda.is_available()): sys.exit()
print("GPU with CUDA support detected. Running CUDA tests...")

### construct input sequence:
in_seq = Variable(torch.randn(batch_size, nchannels, sequence_length).cuda())

### construct residual blocks and fetch outputs:
relu_block = ResidualReLUBlock(nchannels, kernel_width, dilation)
relu_block.init() # (initialize parameters)
relu_block.cuda()
relu_out_seq = relu_block(in_seq)

mu_block = ResidualMUBlock(nchannels, kernel_width, dilation)
mu_block.init() # (initialize parameters)
mu_block.cuda()
mu_out_seq = mu_block(in_seq)

### assertions and prints:
print("===== Assert that ReLU block output dimensions are correct... =====")
print(relu_out_seq.size())
assert (relu_out_seq.size() == in_seq.size())
print("===== Assert that MU block output dimensions are correct... =====")
print(mu_out_seq.size())
assert (mu_out_seq.size() == in_seq.size())

# run overfitting test on constant sequence:
print("===== Attempting to overfit on constant sequences with ReLU block... =====")
source_sequence = Variable(torch.ones(batch_size, nchannels, sequence_length).mul_(2.).cuda())
target_sequence = Variable(torch.ones(batch_size, nchannels, sequence_length).mul_(3.).cuda())
num_iterations = 1000
print_every = 50
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(relu_block.parameters())
for step in range(num_iterations):
    optimizer.zero_grad()
    outseq = relu_block(source_sequence)
    loss = loss_fn(outseq, target_sequence)
    if step % print_every == 0: print("Loss @ step {0}: {1}".format(step,loss.data[0]))
    loss.backward()
    optimizer.step()
print("... Done. You should see a gradually decreasing loss. (If you don't, something went wrong.)")

print("===== Attempting to overfit on constant sequences with MU block... =====")
source_sequence = Variable(torch.ones(batch_size, nchannels, sequence_length).mul_(2.).cuda())
target_sequence = Variable(torch.ones(batch_size, nchannels, sequence_length).mul_(3.).cuda())
num_iterations = 1000
print_every = 50
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mu_block.parameters())
for step in range(num_iterations):
    optimizer.zero_grad()
    outseq = mu_block(source_sequence)
    loss = loss_fn(outseq, target_sequence)
    if step % print_every == 0: print("Loss @ step {0}: {1}".format(step,loss.data[0]))
    loss.backward()
    optimizer.step()
print("... Done. You should see a gradually decreasing loss. (If you don't, something went wrong.)")
