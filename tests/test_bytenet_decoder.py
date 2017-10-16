"""
Training test to make sure the bytenet decoder is capable of learning basic connections.
"""
import torch
from torch.autograd import Variable
import torch.optim as optim
from modules.bytenet_decoder import ByteNetDecoder

### parameters:
nchannels = 7
num_labels = 5
batch_size = 3
encoding_dim = 6
sequence_length = 45
kernel_width = 3
output_dim = 21
layers = [(2,1), (2,2), (2,4), (2,8), (2,16)]
blocktype = 'mult' # or 'relu'

### random input sequences:
in_seq = Variable(torch.randn(batch_size, sequence_length).mul(num_labels).clamp(0,num_labels-1).long())
enc_seq = Variable(torch.randn(batch_size, encoding_dim, sequence_length))

### construct and eval bytenet:
decoder = ByteNetDecoder(num_labels, encoding_dim, nchannels, kernel_width, output_dim, layers, block=blocktype)
next_step_seq = decoder(in_seq,enc_seq)

### asserts and validations:
assert (next_step_seq.size() == torch.Size([batch_size, num_labels, sequence_length]))

### try to overfit on constant sequence (linearly increasing):
print("Attempting to learn a basic monotonically increasing sequence...")
increasing_modulo = torch.arange(0,sequence_length).unsqueeze(0).expand(torch.Size([batch_size,sequence_length])).clamp(0,num_labels-1).long()
source_seq = Variable(increasing_modulo)
target_seq = (source_seq + Variable(torch.ones(source_seq.size()).long())).clamp(0,num_labels-1)
encoding_seq = Variable(torch.randn(batch_size, encoding_dim, sequence_length))
optimizer = optim.Adam(decoder.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

num_iterations = 1000
print_every = 1

for k in range(num_iterations):
    optimizer.zero_grad()
    next_step_preds = decoder(source_seq, encoding_seq)
    loss = 0.
    for t in range(next_step_preds.size(2)):
        loss = loss + loss_fn(next_step_preds[:,:,t], target_seq[:,t])
    loss.backward()
    if k % print_every == 0: print("Step {0} Loss: {1}".format(k,loss.data[0]))
    if loss.data[0] < 1.0: break
    optimizer.step()
print("... Done. You should see a gradually decreasing loss.")
print("(If you don't, it means something went wrong and the model is not learning.)")
