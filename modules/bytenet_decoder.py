"""
The decoder from ByteNet, as described in:

"Neural Machine Translation in Linear Time", N. Kalchbrenner et al,
https://arxiv.org/abs/1610.10099

Credits:
While the usage of the input buffer during incremental forward passes of the
decoder is novel, inspiration was taken from the fairseq architecture by Facebook
Research:
https://github.com/facebookresearch/fairseq-py/blob/master/ \
  fairseq/modules/linearized_convolution.py
"""
import torch
import torch.nn as nn
from modules.block import ResidualReLUBlock, ResidualMUBlock
from collections import OrderedDict

class ByteNetDecoder(nn.Module):
    """
    The ByteNet decoder module is a stack of causal dilated conv1d blocks that takes a batch of
    logits at each timestep, mixes it with an encoded representation, and returns a batch of logits
    representing the predicted next timestep.

    This can be used in two ways: if training against cross entropy loss (with known alignments in
    the target sequence), the sequence can be trained against the whole target sequence to predict
    the next timestep. If training with unknown alignment against a target sequence, we can loop
    linearly over the internal convolutions one-at-a-time until either `max_timesteps` is reached
    or until we observe a <STOP> label.
    """
    def __init__(self, num_labels, encoding_dim, channels, output_dim, layers, block='mult',
                 pad=0, start=5, stop=6, max_timesteps=500):
        """
        Construct all submodules and save parameters.
        
        Args:
        * num_labels: the size of the alphabet (including the NULL/<BLANK> CTC character).
        * encoding_dim: dimension of the output timesteps of the encoded source sequence.
        * channels: the number of channels at each resblock; each tensor in the network will
        have either `channels` or `2*channels` dimensions (depending on the specific sub-module.)
        * output_dim: the dimensionality of the output mapping layers.
        * layers: a python list of integer tuples of the form [(kwidth, dilation)].
        * block: either 'mult' or 'relu'; decides which type of causal ResConv block to use.
        * pad: the padding label (python integer).
        * start: the start label (python integer).
        * stop: stop label (python integer).
        * max_timesteps: don't decode past this number of timesteps.
        """
        super(ByteNetDecoder, self).__init__()
        # save inputs:
        self.num_labels = num_labels
        self.channels = channels
        self.encoding_dim = encoding_dim
        self.output_dim = output_dim
        self.layers = layers
        if not (block in ['relu', 'mult']):
            raise TypeError("The `block` setting must be either `relu` or `mult`.")
        self.block = block
        ResBlock = ResidualMUBlock if (block == 'mult') else ResidualReLUBlock
        self.pad_label = pad
        self.start_label = start
        self.stop_label = stop
        self.max_timesteps = max_timesteps
        
        # construct input embedding and Conv1x1 layer:
        self.input_embed = nn.Embedding(num_labels, 2*channels)
        self.input_conv1x1 = nn.Conv1d(2*channels, 2*channels, kernel_size=1, stride=1, dilation=1)
        
        # conv1x1 to mix in the encoded sequence:
        self.encoding_layer = nn.Conv1d(encoding_dim, 2*channels, kernel_size=1, stride=1, dilation=1)
        
        # stack of causal residual convolutional blocks:
        self.stacked_residual_layer = nn.Sequential(OrderedDict(
            [('resconv{}'.format(l_idx),ResBlock(2*channels, k, dilation=d)) for (l_idx,(k,d)) in enumerate(layers)]
        ))
        
        # Final [Conv1x1=>ReLU=>Conv1x1] mapping to construct outputs:
        self.output_layer = nn.Sequential(
            nn.Conv1d(2*channels, output_dim, kernel_size=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(output_dim, num_labels, kernel_size=1, dilation=1))

        # compute receptive field:
        _rf = 1
        for layer in self.stacked_residual_layer:
            _rf += (layer.receptive_field - 1)
        self.receptive_field = _rf


    def init(self):
        """Initialize params via Kaiming-Normal on weights, noisy zeros on biases."""
        for p in self.parameters():
            if len(p.size()) >= 2: nn.init.kaiming_normal(p)
            if len(p.size()) == 1: p.data.zero_().add(0.0001 * torch.randn(p.size()))


    def linear(self, dec_frames, enc_frames):
        """
        Given an initial collection of input frames, perform evaluation using a linearized version
        of the modules, emitting a single timestep each time this is called.

        **This should NOT be used for training; you should use this for evaluation of a single timestep.**

        Note: we recommend `seq == receptive_field`, as otherwise padding will be automatically applied
        at the start of both sequences until .

        Args:
        * dec_frames ~ LongTensor Variable of shape `(batch, seq)`
        * enc_frames ~ FloatTensor Variable of shape `(batch, encoding_dim, seq)`
        
        Returns:
        * FloatTensor Variable of shape `(batch, num_labels)`. Probability of next label.
        """
        # sanity check:
        #assert (dec_frames.size(1) == enc_frames.size(2))
        # run bytenet stack:
        o = self.input_embed(dec_frames).transpose(1,2) # embed & reshape[BSC=>BCS]
        o = self.input_conv1x1(o)
        o = o + self.encoding_layer(enc_frames)
        o = self.stacked_residual_layer(o)
        o = self.output_layer(o)
        # return final timestep
        return o[:,:,-1]


    def forward(self, encoded_seq, teacher_sequence=None, teacher_ratio=0.5):
        """
        Take an encoded sequence and loop over the dataset, starting with an initial set of frames of
        minimal temporal dimensionality (== receptive_field) given by `[<PAD>] * rf-1 + [<START>]`.
        
        An internal buffer of outputs is maintained.

        Additionally, a `teacher_sequence` can be passed; if not None, then for each timestep t,
        with `probability == teacher_ratio` the value teacher_sequence[t-1] be passed as input.

        Args:
        * encoded_seq: a FloatTensor variable of shape (batch, encoding_dim, encoded_seq_length).
        * teacher_seq: a LongTensor variable of shape (batch, teacher_seq_length). At each timestep less
        than `teacher_seq_length`, we randomly choose between the previous timestep's predicted input
        and the teacher input at this step.
        * teacher_ratio: python float between 0.0 and 1.0 that indicates probability of choosing the
        teacher sequence's input instead of the previous predicted next-step.

        Returns: a tuple (output_sequence, output_lengths) where:
        * `output_sequence` is a FloatTensor variable of shape (batch, num_labels, num_timesteps).
        * `output_lengths`: is an IntTensor variable of shape (batch); contains the lengths of each
        output transcription sequence, from first timestep to the timestep at which <STOP> was emitted.
        """
        # teacher-forcing is currently unimplemented:
        if teacher_sequence is not None: raise TypeError("ERR: teacher forcing is not implemented yet.")

        batch_size = encoded_seq.size(0)
        # create a new buffer ~ (1,receptive_field) LongTensor:
        _buffer = [self.pad_label] * (self.receptive_field-1) + [self.start_label]
        label_buffer = torch.LongTensor(_buffer).view(1,-1).expand(batch_size,self.receptive_field)
        if encoded_seq.is_cuda: label_buffer = label_buffer.cuda() # (load on CUDA if necessary)

        # pad encoded sequence with <PAD> characters:
        encoded_pad = encoded_seq.data.new(batch_size, self.encoding_dim, self.receptive_field-1+encoded_seq.size(2))
        encoded_pad[:,:,(self.receptive_field-1):] = encoded_seq.data.clone()

        # loop until finish:
        output_buffer = []
        num_enc_steps = encoded_seq.size(2)
        output_lengths = torch.zeros(batch_size,2) # [:,0] ~ lengths; [:,1] ~ finished
        if encoded_seq.is_cuda: output_lengths.cuda()
        for k in range(self.max_timesteps):
            # compute output logits for next timestep; use padding labels if no more encoder timesteps:
            if (k < num_enc_steps):
                enc_steps_avail = encoded_pad[:,:,k:(self.receptive_field+k)]
            else:
                enc_steps_avail = encoded_pad.new(batch_size, self.encoding_dim, self.receptive_field).fill_(self.pad_label)
            o = self.linear(torch.autograd.Variable(label_buffer), torch.autograd.Variable(enc_steps_avail))
            
            # append to outputs:
            output_buffer.append(o)

            # compute argmax:
            _, next_label = torch.max(o, dim=1)
            
            # update output_lengths:
            stop_mask = torch.eq(next_label.data, self.stop_label).float()
            output_lengths[:,0].add_(stop_mask).clamp_(max=1) # (flip 'stop' flag)
            output_lengths[:,1].add_(1).sub_(output_lengths[:,0]) # (incr. step-count)
            if torch.eq(output_lengths[:,0],1).all(): break # (end loop if finished)
            
            # shift buffer by 1 step, accommodating the new output:
            label_buffer[:,:-1] = label_buffer[:,1:].clone()
            label_buffer[:,-1] = next_label.data

        # return output steps and sequence lengths:
        return torch.stack(output_buffer, dim=2), torch.autograd.Variable(output_lengths[:,1].int())
