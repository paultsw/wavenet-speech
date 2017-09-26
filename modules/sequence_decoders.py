"""
sequence_decoders.py: functions to convert a sequence of logits to integer strings.
"""
import torch
import torch.nn.functional as F
from modules.beam import Beam


def argmax_decode(logits):
    """
    Given a batch of logits represented as a tensor, 

    Args:
    * logits: a FloatTensor or FloatTensor variable of shape (batch, sequence, logit). The final
    coordinate of the `logit`-dimension is assumed to be the probability of the blank label.
    
    Returns:
    * labels: a LongtTensor or LongTensor variable of shape (batch, sequence), where each entry
    is the integer labeling of the logit, based on the argmaxed coordinate.
    """
    labels = logits.new(logits.size(0), logits.size(1))
    _, labels = torch.max(logits, dim=2)
    return labels


def labels2strings(labels, lookup={0: '', 1: 'A', 2: 'G', 3: 'C', 4: 'T'}):
    """
    Given a batch of labels, convert it to string via integer-to-char lookup table.

    Args:
    * labels: a LongTensor or LongTensor variable of shape (batch, sequence).
    * lookup: a dictionary of integer labels to characters. One of the labels should map to the
    empty string to represent the BLANK label.

    Returns:
    * strings_list: a list of decoded strings.
    """
    if isinstance(labels, torch.autograd.Variable): labels = labels.data
    labels_py_list = [list(labels[k]) for k in range(labels.size(0))]
    strings_list = [ "".join([lookup[ix] for ix in labels_py]) for labels_py in labels_py_list ]
    return strings_list


_DEFAULT_BEAM_MAP_ = { '<pad>': 0, '<s>': 5, '</s>': 6 } # mapping of special symbols required for beam decoder
class BeamSearchDecoder(object):
    """
    A beam search decoder class. Can be called with (e.g.):
    
    >>> decoded_labels = BeamSearchDecoder(beam_width=4)(logits)

    The beams are stored as a list of batch-vectors ~ [(batch_size, beam_width)]
    where each value is an integer label.

    Alongside the beams, the running probabilities are stored as a constant-size
    batch-vector of shape (batch_size, beam_width) indicating the probability of
    each beam for each batch.
    """
    def __init__(self, batch_size, num_labels, mapping_dict=_DEFAULT_BEAM_MAP_, beam_width=5, cuda=False):
        """Store parameters and initialize the beam."""
        self.beam_width = beam_width
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.beams = [Beam(beam_width, mapping_dict, cuda=cuda) for _ in range(batch_size)]


    def decode(self, logits):
        """Decode a batch of logits."""
        # reshape: (batch, num labels, sequence length)=>(sequence length, batch, num labels)
        logits = logits.data.permute(2,0,1)

        # append <START> and <STOP> columns and vectors to logits:
        zero_col = torch.zeros(logits.size(0), logits.size(1), 1)
        logits = torch.cat([logits, zero_col, zero_col], dim=2)
        start_vec = torch.zeros(self.num_labels+2)
        start_vec[self.num_labels] = 1.
        start_vec = start_vec.view(1,1,self.num_labels+2).expand(1, logits.size(1), self.num_labels+2)
        stop_vec = torch.zeros(self.num_labels+2)
        stop_vec[self.num_labels+1] = 1.
        stop_vec = stop_vec.view(1,1,self.num_labels+2).expand(1, logits.size(1), self.num_labels+2)
        print("START:",start_vec.size())
        print(start_vec)
        print("STOP:",stop_vec.size())
        print(stop_vec)
        print("LOGITS:", logits.size())
        logits = torch.cat([start_vec, logits, stop_vec], dim=0)

        # loop through each timestep of logits (after appending <S> & </S>) and update beams:
        for k in range(logits.size(0)):
            label_lkhd = F.softmax( logits[k].view(self.batch_size, logits.size(2)) ).data
            label_lkhd = label_lkhd.unsqueeze(1).expand(self.batch_size, self.beam_width, logits.size(2))
            # update beams:
            for b in range(self.batch_size):
                if self.beams[b].done: continue
                self.beams[b].advance(label_lkhd[b])

        # extract best hypothesis sequences and return along with unnormalized scores:
        hypotheses = []
        probas = []
        for b in range(self.batch_size):
            scores, Ks = self.beams[b].sort_best()
            probas.append( scores[0] )
            hypotheses.append( self.beams[b].get_hyp(Ks[0]) )
        return (probas, hypotheses)
