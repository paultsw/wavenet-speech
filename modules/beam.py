##########################################################################################
# Implementation of a beam search class, borrowed
# from MaximumEntropy on GitHub:
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

class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, beam_width, vocab, cuda=(torch.cuda.is_available())):
        """Initialize params."""
        self.size = beam_width
        self.done = False
        self.pad = vocab['<pad>']
        self.bos = vocab['<s>']
        self.eos = vocab['</s>']
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(beam_width).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(beam_width).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

        # The attentions (matrix) for each time.
        self.attn = []


    def get_current_state(self):
        """
        Get state of beam; get the outputs for the current timestep.
        Return type: FloatTensor ~ (beam_width,).
        """
        return self.nextYs[-1]


    def get_current_origin(self):
        """
        Get the backpointer to the beam for the current timestep.
        Return type: FloatTensor ~ (beam_width,).
        """
        return self.prevKs[-1]


    def advance(self, word_lk):
        """
        Advance the beam.
        
        #  Given prob over words for every last beam `wordLk` and attention
        #   `attnOut`: Compute and update the beam search.
        #
        # Parameters:
        #
        #     * `wordLk`- probs of advancing from the last step (K x words)
        #     * `attnOut`- attention at the last step
        #
        # Returns: True if beam search is complete.
        """
        num_words = word_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
        else:
            beam_lk = word_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True

        return self.done


    def sort_best(self):
        """
        Sort the beam.
        """
        return torch.sort(self.scores, 0, True)


    def get_best(self):
        """
        Get the most likely candidate; get the score of the best in the beam.
        """
        scores, ids = self.sort_best()
        return scores[1], ids[1]


    def get_hyp(self, k):
        """
        Get hypotheses.

        # Walk back to construct the full hypothesis.
        #
        # Parameters.
        #
        #     * `k` - the position in the beam to construct.
        #
        # Returns.
        #
        #     1. The hypothesis
        #     2. The attention at each time step.
        """
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1]
