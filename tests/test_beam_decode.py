"""
An example on usage of OpenNMT's Beam class.
"""
import torch
from modules.beam import Beam

### beam search settings
beam_width = 5
batch_size = 8
target_dict = { '-': 0, 'a': 1, 'g': 2, 'c': 3, 't': 4, '<pad>': 5, '<s>': 6, '</s>': 7 }

### construct a Beam object for each sequence in the batch:
beams = [Beam(beam_width, target_dict, cuda=False) for _ in range(batch_size)]

### generate some example logits:
sequence_length = 44
ndim = 8 # { - A G C T <PAD> <S> </S> }
#logits = torch.nn.functional.normalize(torch.randn(sequence_length, batch_size, ndim).abs_(), p=1, dim=2)
logits = torch.zeros(sequence_length,batch_size,ndim)
for k in range(sequence_length): logits[k,:,2] = 1.

### add <START> and <STOP> tags to the logits:
start_vec = torch.Tensor([0,0,0,0,0,0,1,0]).view(1,1,ndim).expand(1, batch_size, ndim)
stop_vec = torch.Tensor([0,0,0,0,0,0,0,1]).view(1,1,ndim).expand(1, batch_size, ndim)
logits = torch.cat([start_vec, logits, stop_vec], dim=0)
sequence_length += 2

### loop through each timestep and update the respective beams:
for k in range(sequence_length):
    label_lkhd = logits[k].view(batch_size, ndim).transpose(0,1).contiguous()
    label_lkhd = label_lkhd.unsqueeze(1).expand(batch_size,beam_width,ndim)
    # update each beam:
    for b in range(batch_size):
        if beams[b].done: continue
        beams[b].advance(label_lkhd[b])

### return decoded hypthesis sequences and probabilities/scores for each:
num_best = 1 # (only return the top-2 best hypothesis sequence for each batch)
hypotheses = {}
probas = {}
for b in range(batch_size):
    # create dictionary entries for batch sequence b in `hypotheses`, `probas`:
    hypotheses[b] = []
    probas[b] = []
    # get best probabilities and their associated indices in the beam:
    scores, Ks = beams[b].sort_best()
    # append the scores to the list of best probabilities:
    probas[b] += [ scores[0:num_best] ]
    # append the hypothesis sequences to the list of best hypotheses:
    beam_b_hyps = [ beams[b].get_hyp(k) for k in Ks[0:num_best] ]
    hypotheses[b] += beam_b_hyps

### print outputs:
for k in range(batch_size):
    print("=" * 80)
    print("Logits:")
    print(logits[:,k,:])
    print("Hypotheses, batch sequence {}".format(k))
    print(hypotheses[k])
    print("Probabilities, batch sequence {}".format(k))
    print(probas[k])
