##########################################################################################
# Implementation of a beam search class, edited & borrowed from MaximumEntropy on GitHub:
#
#   https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/beam_search.py
#
# who, in turn, borrowed it from OpenNMT's PyTorch implementation:
#
#   https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
#
# For usage example, see `wavenet-speech/tests/test_beam_decode.py`
##########################################################################################
import torch

# True iff CUDA is auto-detected
__CUDA_AVAIL__ = torch.cuda.is_available()

class Beam(object):
    """
    The Beam object contains an ordered beam of candidate hypothesis sequences;
    it is sequentially updated at each logit step.

    (Does *NOT* operate on batches; you need to instantiate a Beam object for each sequence in a batch.)
    """
    def __init__(self, beam_width, mapping_dict, cuda=__CUDA_AVAIL__):
        """
        Store parameters and initialize probability scores, backpointers, 
        """
        self.beam_width = beam_width
        self.done = False
        self.pad_label = mapping_dict['<pad>']
        self.start_label = mapping_dict['<s>']
        self.end_label = mapping_dict['</s>']
        
        # initialize probability scores to 0 for each beam; optionally place on CUDA:
        self.scores = torch.FloatTensor(beam_width).zero_()
        if cuda: self.scores = self.scores.cuda()
        
        # init backpointers as empty list (since no backptrs at initial step):
        self.prev_Ks = []
        
        # init next step as pad labels, except for the first beam hypothesis (to break symmetry):
        _init_outputs = torch.LongTensor(beam_width).fill_(self.pad_label)
        if cuda: _init_outputs = _init_outputs.cuda()
        self.next_Ys = [_init_outputs]
        self.next_Ys[0][0] = self.start_label

        # attention at each timestep:
        self.attns = []


    def get_current_state(self):
        """
        Get the outputs for the current timestep.
        Returns FloatTensor of shape (beam_width).
        """
        return self.next_Ys[-1]

    
    def get_current_origin(self):
        """
        Get the backpointer to the previous timestep.
        Returns FloatTensor of shape (beam_width).
        """
        return self.prev_Ks[-1]


    def sort_best(self):
        """
        Sort the probability scores of the beams.
        """
        return torch.sort(self.scores, dim=0, descending=True)


    def get_best(self):
        """
        Get the most likely candidate hypothesis sequence and its score.
        """
        scores, id_seqs = self.sort_best()
        return (scores[0], id_seqs[0])


    def get_hyp(self, k):
        """
        Get hypothesis sequences by walking backwards through the backpointers.
        
        Params:
        * k: the position in the beam to construct, i.e. return beam k.
        Satisfies `0 <= k < beam_width`.
        """
        hypothesis_seq = []
        # backwards loop over backpointers:
        for j in range(len(self.prev_Ks)-1, -1, -1):
            hypothesis_seq.append(self.next_Ys[j+1][k])
            k = self.prev_Ks[j][k]
        return hypothesis_seq[::-1] # (fancy indexing reverses the list)


    def advance(self, label_dist):
        """
        Advance the beam by taking in a distribution over the next label.

        Params:
        * label_dist: a FloatTensor of shape (beam_width, num_labels) containing the probabilities
        over the next label, for each hypothesis sequence in the beam.
        Returns:
        * True if this beam is done processing (encountered a STOP label); False otherwise.
        """
        # get number of labels (for convenience):
        num_labels = label_dist.size(1)
        
        # add previous score to beam-probability if not first timestep:
        if len(self.prev_Ks) > 0:
            beam_dist = label_dist + self.scores.unsqueeze(1).expand_as(label_dist)
        else:
            beam_dist = label_dist[0]

        # get the most probable next labels and update running probability scores:
        flat_beam_dist = beam_dist.view(-1)
        best_scores, best_scores_ids = flat_beam_dist.topk(self.beam_width, dim=0, largest=True, sorted=True)
        self.scores = best_scores

        # `best_scores_ids` is a flattened LongTensor of shape (beam_width*num_labels), so we have to
        # re-calculate which label & beam each score came from:
        prev_k = (best_scores_ids / num_labels)
        self.prev_Ks.append(prev_k) # update backpointers
        self.next_Ys.append(best_scores_ids - prev_k * num_labels) # update output labels
        
        # check if we've reached EOS and set `done` flag if true:
        if (self.next_Ys[-1][0] == self.end_label): self.done = True
        return self.done
