"""
Tests for WaveNet-Speech convolutional transcription network.
"""
import torch
from torch.autograd import Variable
from modules.classifier import WaveNetClassifier
from warpctc_pytorch import CTCLoss

### constructor parameters:
num_labels = 8
batch_size = 3
seq_dim = 256
seq_length = 10000
output_dim = 128
dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
layers = [(seq_dim, seq_dim, 2, d) for d in dilations]
downsample_rate = 3

### construct fake data:
input_seq = Variable(torch.randn(batch_size, seq_dim, seq_length))

### construct wavenet classifier/transcriptor stack & run:
# (this time we /DO/ have to softmax the outputs, unlike previously)
classifier = WaveNetClassifier(seq_dim, num_labels, layers, output_dim,
                               pool_kernel_size=downsample_rate, softmax=True)
out_dist_seq = classifier(input_seq)

### print outputs, sizes, etc:
print("Downsample rate:")
print(downsample_rate)
print("Input Sequence Size:")
print(input_seq.size())
print("Output Sequence Size:")
print(out_dist_seq.size())
#print(out_dist_seq)

assert (batch_size == out_dist_seq.size(0))
assert (num_labels == out_dist_seq.size(1))

# overfit on single pair of sequences, batch size == 1:
print("===== Attempting to overfit on a constant pair of sequences (with CTC loss)... =====")
source_seq = Variable(torch.randn(1, seq_dim, seq_length))
print("* Source sequence:")
print(source_seq)
target_seq = Variable(torch.zeros(int(seq_length / 4)).fill_(num_labels-2).int())
target_seq[0:50] = 1
print("* Target sequence:")
print(target_seq)
loss_fn = CTCLoss()
label_sizes = Variable(torch.IntTensor([target_seq.size(0)]))
num_iterations = 1000
print_every = 1
optimizer = torch.optim.RMSprop(classifier.parameters())
print("Training...")
try:
    for k in range(num_iterations):
        optimizer.zero_grad()
        ctc_pred = classifier(source_seq)
        transcriptions = ctc_pred.permute(2,0,1).contiguous()
        transcription_sizes = Variable(torch.IntTensor([ctc_pred.size(2)]))
        loss = loss_fn(transcriptions, target_seq, transcription_sizes, label_sizes)
        loss.backward()
        optimizer.step()
        if k % print_every == 0: print("Loss @ step {0}: {1}".format(k, loss.data[0]))
except KeyboardInterrupt:
    print("...Halted training.")
finally:
    print("... Done. You should see a gradually decreasing loss. (If you don't, it means that something went wrong.)")
    print("Final predicted values:")
    print(ctc_pred.data)
