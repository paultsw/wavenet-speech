"""
Training test to make sure the bytenet decoder is capable of learning basic connections.
"""
import torch
from torch.autograd import Variable
import torch.optim as optim
from modules.bytenet_decoder import ByteNetDecoder
from warpctc_pytorch import CTCLoss

_CUDA_ = torch.cuda.is_available()

### parameters:
num_labels = 8
encoding_dim = 8
num_channels = 12
batch_size = 3
output_dim = 21
layers = [(3,1), (3,2), (3,4), (3,8), (3,16)] * 2
blocktype = 'mult' # or 'relu'
max_steps = 50

# labels: 1-4 ~ nucleotides, 0 = pad/blank, 5 = <START>, 6 = <STOP>
pad_label = 7
start_label = 5
stop_label = 6


### construct  bytenet decoder:
decoder = ByteNetDecoder(num_labels, encoding_dim, num_channels, output_dim, layers, block=blocktype,
                         pad=pad_label, start=start_label, stop=stop_label, max_timesteps=max_steps)

### evaluate on random input/encoding timesteps:
inp_frames = Variable(torch.randn(batch_size, decoder.receptive_field).mul(num_labels).clamp(0,num_labels-1).long())
enc_frames = Variable(torch.randn(batch_size, encoding_dim, decoder.receptive_field))
out_frame = decoder.linear(inp_frames, enc_frames)


### asserts and validations:
assert (out_frame.size() == torch.Size([batch_size, num_labels]))
print(out_frame) # [Comment/Un-Comment this as necessary]


### evaluate on forward loop:
encoder_seq_length = 50
encoder_seq = Variable(torch.randn(batch_size, encoding_dim, encoder_seq_length))
out_seq, out_lengths = decoder(encoder_seq)
print("Encoded sequence size: {}".format(encoder_seq.size()))
print("Output sequence size: {}".format(out_seq.size()))
print(out_seq) # [Comment/Un-Comment this as necessary]
print(out_lengths)


### try to overfit on constant sequence (linearly increasing, batch size == 1):
print("Attempting to learn a basic monotonically increasing sequence...")
input("[ENTER to begin]")
if _CUDA_: decoder.cuda()
incr_ = torch.IntTensor(([1,2,3,4] * 5) + [6]).view(1,-1)
target_seq = Variable(incr_)
target_seq_length = Variable(torch.IntTensor([target_seq.size(1)])) # (num target timesteps + <STOP>)
encoding_seq = Variable(torch.randn(1,encoding_dim, 100))
if _CUDA_: encoding_seq = encoding_seq.cuda()
optimizer = optim.Adamax(decoder.parameters())
loss_fn = CTCLoss()

num_iterations = 10000
print_every = 25
for k in range(num_iterations):
    optimizer.zero_grad()
    out_seq, out_lengths = decoder(encoding_seq) # out_seq ~ BCS
    transcription = out_seq.permute(2,0,1).cpu() # reshape BCS => SBC
    loss = loss_fn(transcription, target_seq.cpu(), out_lengths.cpu(), target_seq_length.cpu())
    loss.backward()
    if k % print_every == 0: print("Step {0} Loss: {1}".format(k,loss.data[0]))
    if loss.data[0] < 1.0: break
    optimizer.step()
print("... Done. You should see a gradually decreasing loss.")
print("(If you don't, it means something went wrong and the model is not learning.)")
