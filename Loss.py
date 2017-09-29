import torch.nn as nn
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

class Loss(object):
    """Wrapper class for loss functions."""
    def __init__(self, loss_choice, ce_weights=None, joint_balance=None, averaged=True):
        """Construct a loss function wrapper."""
        assert (loss_choice in ['joint','ctc'])
        self.loss_choice = loss_choice
        if loss_choice == 'joint':
            self.xe_loss_fn = nn.CrossEntropyLoss(weights=ce_weights)
            self.ctc_loss_fn = CTCLoss()
        if loss_choice == 'ctc':
            self.ctc_loss_fn = CTCLoss()
        self.averaged = averaged

    def calculate(self, signal, signal_pred, transcription_seq, target_seq, target_lengths, avg=True):
        """
        Returns loss values after computing.

        If loss choice is 'joint', return a tuple `(xe_loss, ctc_loss)`.
        If loss choice is 'ctc', returns the tuple `(None, ctc_loss)`.

        Args:
        * signal: the input signal; this is ignored if `(loss_choice == ctc)`.
        * signal_pred: the predicted reconstruction of the next timestep of the signal; this is
        ignored if `(loss_choice == ctc)`.
        * transcription_seq: the predicted transcription of the CTC network.
        * target_seq: the flattened target labels.
        * target_lengths: a vector of lengths for the target sequence.
        
        *** N.B.: the target sequence should reserve the `0` label for CTC blanks; this method
        does not take care of this for you, so it should be done before calling this method,
        inside the main training loop.
        """
        # optionally compute cross entropy loss:
        if self.loss_choice == 'joint':
            xe_loss = 0.
            _, dense_signal = torch.max(signal[:,:,1:], dim=1)
            for t in range(signal_pred.size(2)-1):
                xe_loss = xe_loss + self.xe_loss_fn(signal_pred[:,:,t], dense_signal[:,t])
            avg_xe_loss = xe_loss / signal.size(2)
        else:
            xe_loss = None
            avg_xe_loss = None
        
        # compute CTC loss:
        transcriptions = transcription_seq.permute(2,0,1).contiguous() # reshape to (seq, batch, logits)
        transcription_lengths = Variable(torch.IntTensor([transcriptions.size(0)] * transcriptions.size(1)))
        labels = target_seq.cpu().int()
        ctc_loss = self.ctc_loss_fn(transcriptions, labels, transcription_lengths, target_lengths)
        avg_ctc_loss = ctc_loss / transcriptions.size(0)
        
        if self.averaged:
            return (avg_xe_loss, avg_ctc_loss)
        else:
            return (xe_loss, ctc_loss)
