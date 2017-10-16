"""
Learning test for Multiplicative Unit.
"""
import torch
from torch.autograd import Variable
from modules.block import MultiplicativeUnit
import torch.optim as optim

### params:
ndim = 7
kernel_width = 3
dilation = 2
batch_size = 5
sequence_length = 12

### construct input seq, multiplicative block, output seq:
in_seq = Variable(torch.randn(batch_size, ndim, sequence_length))
munit = MultiplicativeUnit(ndim, kernel_width, dilation=dilation)
munit.init() # (initialize parameters)
out_seq = munit(in_seq)

### assertions, prints:
in_shape = in_seq.size()
out_shape = out_seq.size()
assert (in_shape == out_shape)
print("Input sequence shape: {}".format(in_shape))
print("Output sequence shape: {}".format(out_shape))

### overfitting test:
print("===== Attempting to overfit on constant sequences with MultiplicativeUnit... =====")
source_sequence = Variable(torch.ones(batch_size, ndim, sequence_length).mul_(2.))
target_sequence = Variable(torch.ones(batch_size, ndim, sequence_length).mul_(3.))
num_iterations = 1000
print_every = 50
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(munit.parameters())
for step in range(num_iterations):
    optimizer.zero_grad()
    out = munit(source_sequence)
    loss = loss_fn(out, target_sequence)
    if step % print_every == 0: print("Loss @ step {0}: {1}".format(step,loss.data[0]))
    loss.backward()
    optimizer.step()
print("... Done. You should see a gradually decreasing loss. (If you don't, something went wrong.)")
