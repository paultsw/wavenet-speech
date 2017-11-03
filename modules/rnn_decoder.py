"""
A thin RNN decoder that can be placed on top of the deep convolutional encoder.

Offers a nearly identical interface to the ByteNetDecoder.

(This is a bit of a hack until a convolutional decoder can be implemented with low
memory usage.)
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class StackedLSTMCell(nn.Module):
    """
    A looping stack of LSTM cells, with FC()=>ELU() linking between them, as well as both residual
    connections that additively hop over each RNN block.
    
    Between each layer, we have a fully-connected [hidden_dim=>hidden_dim] layer that transforms the
    hidden state into the input for the next layer.
    """
    def __init__(self, hidden_dim, num_layers):
        """
        Construct a stacked LSTM cell.
        """
        # parent init:
        super(StackedLSTMCell, self).__init__()
        # save params:
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # construct modules:
        self.lstm_cells = nn.ModuleList([nn.LSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.fc_layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU()) for _ in range(num_layers)])
        self.skip_layers = nn.ModuleList([nn.Linear(hidden_dim,hidden_dim) for _ in range(num_layers)])
        # initialize sensibly:
        for p in self.lstm_cells.parameters():
            if len(p.size()) > 1: 
                nn.init.xavier_normal(p)
            else:
                p.data.zero_().add(torch.randn(p.size()).mul(0.001))
        for p in self.fc_layers.parameters():
            if len(p.size()) > 1: 
                nn.init.xavier_normal(p)
            else:
                p.data.zero_().add(torch.randn(p.size()).mul(0.001))
        for p in self.skip_layers.parameters():
            if len(p.size()) == 2: 
                nn.init.eye(p)
            else:
                p.data.zero_().add(torch.randn(p.size()).mul(0.00001))


    def forward(self, x, h0s, c0s):
        """
        Run over one iteration.
        
        Args:
        * x: the bottom-most input.
        * h0s: ...
        * c0s: ...
        Returns:
        * outs: an output FloatTensor Variables of shape (batch_size, hidden_dim); this is the linearized
        sum of the 
        * h1s: a list of the form (h01,h02,h03...) of dtype FloatTensor and shape (batch_size, hidden_dim).
        * c1s: same dtype/shapes as h0s.
        """
        # sanity checks
        assert (len(c0s) == self.num_layers)
        assert (len(h0s) == self.num_layers)
        # compute residual layer and append skip connection to `outs` list
        h1s = []
        c1s = []
        outs = []
        out = x
        for l in range(self.num_layers):
            h1,c1 = self.lstm_cells[l](out, (h0s[l],c0s[l]) )
            out = self.fc_layers[l](h1) + out
            h1s.append(h1)
            c1s.append(c1)
            outs.append(self.skip_layers[l](out))
        # compute sum of skip connections:
        skip_out = torch.sum(torch.stack(outs,dim=0), dim=0)
        # return:
        return (skip_out, h1s, c1s)


class RNNByteNetDecoder(nn.Module):
    """
    RNN-based ByteNet-style decoder. Intended to have a lower memory footprint than the
    convolutional ByteNet decoder and to be a proof-of-concept demo for the overall ByteNet
    architecture.
    """
    def __init__(self, num_labels, encoding_dim, hidden_dim, out_dim, num_layers,
                 pad=0, start=5, stop=6, max_timesteps=500):
        """
        Construct all submodules and save parameters.
        
        Args:
        * num_labels: number of labels in the output label class, including blank/stop/start.
        * encoding_dim: the dimension of the encoder's output timesteps.
        * hidden_dim: the hidden dimension of the inner LSTM cells.
        * out_dim: the hidden dimension of the output FC layer.
        * num_layers: the number of layers in the internal stacked LSTM cell.
        * pad: the BLANK label. (Doubles as a PAD label.)
        * start: the START label.
        * stop: the STOP label.
        * max_timesteps: maximum number of timesteps to run the LSTM cell until we call a stop.
        """
        # parent init:
        super(RNNByteNetDecoder, self).__init__()

        # save params:
        self.num_labels = num_labels
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.pad_label = pad
        self.start_label = start
        self.stop_label = stop
        self.max_timesteps = max_timesteps

        # input layer, embedding:
        self.input_layer = nn.Sequential(
            nn.Embedding(num_labels, encoding_dim),
            nn.Linear(encoding_dim, hidden_dim))

        # encoding layer: mix input and encoding together with FC:
        self.encoder_layer = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim),
            nn.ELU(),
            nn.Linear(encoding_dim, hidden_dim))

        # stack of inner LSTM cells:
        self.lstm_stack = StackedLSTMCell(hidden_dim, num_layers)
        
        # output FC layer:
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ELU(),
            nn.Linear(out_dim, num_labels))

        # perform initializations:
        # [don't init `self.lstm_stack`; already done above]
        for p in self.input_layer.parameters():
            if (len(p.size()) > 1):
                nn.init.xavier_normal(p)
            else:
                p.data.zero_().add(torch.randn(p.size()).mul(0.0001))
        for p in self.encoder_layer.parameters():
            if (len(p.size()) > 1):
                nn.init.xavier_normal(p)
            else:
                p.data.zero_().add(torch.randn(p.size()).mul(0.0001))
        for p in self.output_layer.parameters():
            if (len(p.size()) > 1):
                nn.init.xavier_normal(p)
            else:
                p.data.zero_().add(torch.randn(p.size()).mul(0.0001))


    def forward(self, x0, hvals, cvals, enc_step):
        """
        One pass of the decoder.

        Args:
        * x0: a LongTensor Variable of size (batch_size,). The input at this timestep.
        * hvals: a floattensor variable list of type [(batch_size, hidden_dim)] providing the hidden state
        vectors for each layer of the stack at this timestep.
        * cvals: same type/shape as hvals; the cell state vectors at this timestep.
        * enc_step: a FloatTensor Variable of shape (batch_size, encoding_dim). The encoding vector
        at this timestep.

        Returns: a tuple (out, hvals, cvals) where:
        * out: a floattensor variable of shape `(batch_size,num_labels)` giving a distribution
        over the output labels at this timestep.
        * cvals: a list of the same shape and type as the `cvals` argument.
        * hvals: same type/shape as above.
        """
        out = self.input_layer(x0) + self.encoder_layer(enc_step)
        out, hvals, cvals = self.lstm_stack(out, hvals, cvals)
        out = self.output_layer(out)
        return (out, hvals, cvals)

    def unfold(self, encoding_seq):
        """
        Take an encoding seq and and loop (starting from <START>) until we observe a <STOP> label.

        Args:
        * x0: a LongTensor Variable of size (batch_size,). The first timestep.
        * encoding_seq: a FloatTensor Variable of shape (batch_size, encoding_dim, seq).
        The encoding sequence will be reshaped to (seq, batch_size, encoding_dim) for efficiency;
        the initial expected shape is presumed due to the convolutional encoder.

        Returns: a tuple (outs, lengths) where:
        * outs: a FloatTensor Variable of shape (seq, batch_size, num_labels);
        * lengths: an IntTensor of shape (batch_size) indicating the length of each sequence. (NOT Variable.)
        """
        # formatting, etc.:
        enc = encoding_seq.permute(2,0,1) # ~ reshape => (seq x batch x encoding_dim)
        batch_size = enc.size(1)
        out = Variable(make_longtensor([self.start_label] * batch_size, cuda=enc.is_cuda)) # ~ (batch_size,) Long of <START>s
        shape = torch.Size([batch_size, self.hidden_dim]) # ~ (batch x hidden)
        hvals = [Variable(make_randn(shape, cuda=enc.is_cuda)) for _ in range(self.num_layers)]
        cvals = [Variable(make_randn(shape, cuda=enc.is_cuda)) for _ in range(self.num_layers)]
        num_enc_steps = enc.size(0)
        logits_seq = []
        output_lengths = enc.data.new(batch_size,2).zero_() # [:,0] ~ lengths; [:,1] ~ finished
        enc_zeros_pad = enc[0].data.new(enc[0].size()).zero_()

        # loop through the internal submodules until all finished:
        for t in range(self.max_timesteps):
            # compute logits at this timestep: (have_timesteps ? enc[t] : broadcast 0's)
            enc_step = enc[t] if (t < num_enc_steps) else Variable(enc_zeros_pad)
            logits, hvals, cvals = self.forward(out, hvals, cvals, enc_step)
            logits_seq.append(logits)
            
            # compute label at this timestep:
            _, out = torch.max(logits, dim=1)

            # update `output_lengths`:
            stop_mask = torch.eq(out.data, self.stop_label).float()
            output_lengths[:,0].add_(stop_mask).clamp_(max=1) # flip <STOP> flag
            output_lengths[:,1].add_(1).sub_(output_lengths[:,0]) # incr. step-count

            # quit if all finished early:
            if torch.eq(output_lengths[:,0],1).all(): break

        return ( torch.stack(logits_seq, dim=0), output_lengths[:,1].int() )


### HELPER FUNCTIONS
def make_longtensor(val, cuda=False):
    """Construct longtensor on either CUDA or CPU."""
    if cuda:
        return torch.cuda.LongTensor(val)
    else:
        return torch.LongTensor(val)

def make_randn(shape, cuda=False):
    """Construct a small random value of some shape, either on CUDA or CPU."""
    if cuda:
        return torch.randn(shape).mul_(0.001).cuda()
    else:
        return torch.randn(shape).mul_(0.001)
