"""
sequence_decoders.py: functions to convert a sequence of logits to integer strings.
"""
import torch
import torch.nn.functional as F


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


def labels2strings(labels, lookup={0: 'A', 1: 'G', 2: 'C', 3: 'T', 4: ''}):
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
    def __init__(self, batch_size, num_labels, beam_width=5):
        """Store parameters and initialize the beam."""
        self.beam_width = beam_width
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.beams = []
        self.probas = torch.ones(batch_size, beam_width)


    def __call__(self, logits):
        """
        Scan through the list of logits one timestep at a time and update the beam.
        
        `logits` is expected to be of shape (batch_size, sequence_length, num_labels).
        """
        for tt in range(logits.size(1)): self.beam_search_step(logits[:,tt,:])
        return self.format_beam_output()


    def beam_search_step(self, logit_step):
        """
        Given a vector representing logits at one timestep, update the internal beams
        and probabilities.

        `logit_step` is assumed to be a FloatTensor of shape (batch_size, num_labels).

        * [TODO: optimize this step; TopK might be expensive?]
        """
        # outer-product the current probabilities so-far ~ (batch_size, beam_width) with
        # the logits for each label ~ (batch_size, num_labels):
        # outer_prod ~ (batch_size, beam_width, num_labels)
        outer_prod = self.probas.unsqueeze(2) * logit_step.unsqueeze(1)

        # reduce the outer product with a TopK to get most likely beam candidates:
        topk_probas, topk_ixs = torch.topk(
            outer_prod.view(self.batch_size, self.beam_width * self.num_labels), self.beam_width, dim=1)
        new_probas, new_labels = self.translate_topk(topk_probas, topk_ixs)
        
        # append the new timestep beams to `self.beams` and update probas:
        self.beams.append(new_labels)
        self.probas = new_probas


    def translate_topk(self, vals, ixs):
        """
        Rearrange the results of a flattened Top-K operation on a tensor of shape
        (batch_size, beam_width * num_labels).
        `vals` and `ixs` are both assumed to be of shape (batch_size, beam_width).

        Returns a tuple (probas, labels) where:
        * probas ~ (batch_size, beam_width) is the updated probability for each beam.
        * labels ~ (batch_size, beam_width) is the label to append to each beam.
        """
        sorted_vals, sort_ixs = torch.sort(vals, dim=1)
        pass # [TBD]


    def format_beam_output(self):
        """
        Prepare `self.beams` & `self.probas` for output; return a tuple of decoded
        sequences and probabilities for each candidate.
        """
        stitched_beams = torch.stack(self.beams, dim=2) # [check to make sure this is correct]
        return stitched_beams, self.probas
