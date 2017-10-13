"""
Fully Convolutional Decoder.
"""
import torch
import torch.nn as nn
import torch.nn.init as nn_init
from torch.autograd import Variable
import torch.nn.functional as F
from math import floor, ceil

__CUDA__ = torch.cuda.is_available()

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6, use_cuda=False):
        super(LayerNorm, self).__init__()
        self.use_cuda = use_cuda
        self.gamma = nn.Parameter(torch.ones(features)).unsqueeze(0).unsqueeze(1)
        self.beta = nn.Parameter(torch.zeros(features)).unsqueeze(0).unsqueeze(1)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1).expand_as(x)
        std = x.std(-1).expand_as(x)
        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps) + self.beta.expand_as(x)

    def cuda(self):
        self.gamma = self.gamma.cuda()
        self.beta = self.beta.cuda()


class CausalConv1d(nn.Module):
    """
    A causal 1D convolution with optional dilation.
    """
    def __init__(self, kernel_size, in_channels, out_channels, dilation=1, use_cuda=__CUDA__):
        """
        Construct causal 1d convolution layer.
        """
        super(CausalConv1d, self).__init__()
        
        # attributes:
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.dilation = dilation
        
        # modules:
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels,
                                      kernel_size, stride=1,
                                      padding=(kernel_size-1),
                                      dilation=dilation)

        if use_cuda: self.conv1d.cuda()

    def forward(self, seq):
        """
        Note that Conv1d expects (batch, in_channels, in_length).
        We assume that seq ~ (len(seq), batch, in_channels), so we'll reshape it first.
        """
        seq_ = seq.permute(1,2,0)
        conv1d_out = self.conv1d(seq_).permute(2,0,1)
        # remove k-1 values from the end by taking only the first |seq| entries:
        return conv1d_out[0:len(seq)]
        

class AttnConvolutionalDecoder(nn.Module):
    """
    A fully-convolutional decoder with attention.
    """
    def __init__(self, num_labels, embed_dim, layers, encoding_dim, batch_size, max_time, norm=False, use_cuda=__CUDA__):
        """
        Construct a new attentional convolutional decoder.

        Args:
        * num_labels: number of labels in the target space.
        * embed_dim: dimensionality of the latent dimension to project
        * layers: description of convolutional layers as a list [(kwidth, in_channels, out_channels)].
        * encoding_size:
        * batch_size:
        * max_time: the number of timesteps in the decoded sequence.
        * norm: if True, apply a LayerNorm before entering each residual conv block.
        * use_cuda:
        """
        ### run parent constructor:
        super(AttnConvolutionalDecoder, self).__init__()

        ### store attributes:
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.layers = layers
        self.encoding_dim = encoding_dim
        self.batch_size = batch_size
        self.max_time = max_time
        self.norm = norm
        self.use_cuda = use_cuda

        ### construct all layers:
        # embedding from labels to latent dimension:
        self.label_embed_fn = nn.Embedding(num_labels, embed_dim)
        # embeddings from timestep to latent dimension:
        self.time_embed_fn = nn.Embedding(max_time, embed_dim)
        # sanity check to make sure conv_layers formatted properly:
        assert (embed_dim == layers[0][1])

        # convolutional blocks:
        if norm: norms = []
        residuals = []
        convs = []
        attns = []
        for (kwidth, in_chs, out_chs) in layers:
            if norm: norms.append(LayerNorm(in_chs, use_cuda=use_cuda))
            residuals.append(nn.Linear(embed_dim, out_chs))
            convs.append(CausalResidualConvBlock(batch_size, in_chs, out_chs, kwidth, stride=1, use_cuda=use_cuda))
            attns.append(AttnLayer(out_chs, encoding_dim, embed_dim, batch_size, use_cuda=use_cuda))
        if norm: self.norms = nn.ModuleList(norms)
        self.input_residuals = nn.ModuleList(residuals)
        self.convs = nn.ModuleList(convs)
        self.attns = nn.ModuleList(attns)
        # sanity check:
        assert (len(self.convs) == len(self.attns) and len(layers) == len(self.convs))

        # output projection to label space:
        # (gets softmaxed in the training loop to generate a distribution)
        self.output_residual = nn.Linear(embed_dim, num_labels)
        self.output_proj = nn.Linear(layers[-1][2], num_labels)

        # place weights on CUDA if necessary:
        if use_cuda:
            self.label_embed_fn.cuda()
            self.time_embed_fn.cuda()
            [res.cuda() for res in self.input_residuals]
            self.output_residual.cuda()
            self.output_proj.cuda()
            if norm: [m.cuda() for m in self.norms]


    def init_params(self):
        """
        Sensible initializations, as per the FAIRSeq paper.
        """
        # embeddings:
        nn_init.xavier_normal(self.label_embed_fn.weight)
        nn_init.xavier_normal(self.time_embed_fn.weight)
        # outputs:
        nn_init.xavier_normal(self.output_residual.weight)
        self.output_residual.bias.data.zero_()
        nn_init.xavier_normal(self.output_proj.weight)
        self.output_proj.bias.data.zero_()
        # input residuals:
        for resmod in self.input_residuals:
            nn_init.xavier_normal(resmod.weight)
            resmod.bias.data.zero_()
        # conv blocks:
        [block.init_params() for block in self.convs]
        # attn layers:
        [attn.init_params() for attn in self.attns]


    def forward(self, encoding, targets):
        """
        Given an encoded sequence and a target sequence, predict the target sequence shifted by one timestep.

        `target` is assumed to be T[0:len(T)-1] for some T ~ full target sequence; the goal is to predict
        T[1:len(T)], i.e. each output should be the next timestep prediction.

        This is unusable for anything except training; to generate predictions from scratch, you should save
        this model and use the weights of each submodule on a looping decoder. The reason the forward() pass
        is designed this way is to provide us with the benefits of parallelism during training time.

        Args:
        * encoding: FloatTensor variable of shape (len(encoding), batch, encoding_dim).
        * targets: LongTensor variable of shape (max_time, batch).

        Returns:
        * outputs: output sequence; a FloatTensor of shape (max_time, batch, num_labels). The un-normalized
          probabilities for each label.
        """
        # embed target sequence into latent dimension:
        embedded_targets = self.label_embed_fn(targets.view(self.max_time * self.batch_size)).view(
            self.max_time, self.batch_size, self.embed_dim)

        # construct position embeddings:
        positions = torch.arange(0,self.max_time).unsqueeze(1).expand(self.max_time, self.batch_size).long().contiguous()
        if self.use_cuda: positions = positions.cuda()
        positions = Variable(positions)
        embedded_positions = self.time_embed_fn(positions.view(-1)).view(
            self.max_time, self.batch_size, self.embed_dim)

        # append positions to targets:
        embedded_targets = embedded_targets + embedded_positions

        # pass through a norm=>conv=>attn block for each layer:
        conv_seq = embedded_targets
        for ll in range(len(self.layers)):
            # optional layer norm:
            if self.norm: conv_seq = self.norms[ll](conv_seq)
            # causal-residual convolution block:
            _residual = self.input_residuals[ll](embedded_targets.view(-1, self.embed_dim)).view(
                self.max_time, self.batch_size, self.layers[ll][2])
            conv_seq = self.convs[ll](conv_seq) + _residual
            # add context from attention:
            conv_seq = conv_seq + self.attns[ll](conv_seq, encoding, embedded_targets)

        # output projection and return:
        outputs = self.output_proj(conv_seq.view(-1, conv_seq.size(2))).view(
            self.max_time, self.batch_size, self.num_labels)
        outputs = outputs + self.output_residual(embedded_targets.view(-1, self.embed_dim)).view(
                self.max_time, self.batch_size, self.num_labels)

        return outputs


# ===== ===== ===== ===== Residual Convolutional Block with GLU Activations and causal convolutions
class CausalResidualConvBlock(nn.Module):
    """
    A convolutional block with residual connections.

    (At this current moment, we only support stride == 1 due to the complications of padding calculations.)
    """
    def __init__(self, batch_size, in_channels, out_channels, kwidth, stride=1, use_cuda=__CUDA__):
        """
        Construct a new causal ResConvBlock.
        """
        # run parent constructor:
        super(CausalResidualConvBlock, self).__init__()

        # raise error if stride != 1:
        if stride != 1: raise Exception("[NANOSEQ] ERR: strides =/= 1 are not supported.")

        # store attributes:
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kwidth = kwidth
        self.stride = stride
        self.use_cuda = use_cuda

        # construct submodules:
        self.conv1d_glu = CausalConv1d(kwidth, in_channels, out_channels, use_cuda=use_cuda)
        self.conv1d_id = CausalConv1d(kwidth, in_channels, out_channels, use_cuda=use_cuda)
        self.residual_proj = nn.Linear(in_channels, out_channels)
        self.glu = GLU()

        # move to CUDA if necessary:
        if use_cuda: self.residual_proj.cuda()


    def forward(self, in_seq):
        """
        Forward pass through conv block. Reshape first to (batch, in_depth, seq).

        Args:
        * in_seq: input sequence of shape (seq, batch, in_depth). Pass the sequence through
          convolution blocks and end with a GLU.

        Returns:
        * out_seq: sequence of shape (seq, batch, out_depth).
        """
        conv_glu = self.conv1d_glu(in_seq)
        conv_id = self.conv1d_id(in_seq)

        # apply GLU() and add input sequence:
        glu_out = self.glu(conv_glu, conv_id)
        in_seq_proj = self.residual_proj(in_seq.view(-1, self.in_channels)).view(
            len(in_seq), self.batch_size, self.out_channels)
        return torch.add(glu_out, in_seq_proj)


    def init_params(self):
        """
        Initialize all parameters.
        """
        ### Conv1d-GLU:
        nn_init.xavier_normal(self.conv1d_glu.conv1d.weight)
        self.conv1d_glu.conv1d.bias.data.zero_()

        ### Conv1d-Id:
        nn_init.xavier_normal(self.conv1d_id.conv1d.weight)
        self.conv1d_id.conv1d.bias.data.zero_()

        ### linear residual projection:
        nn_init.xavier_normal(self.residual_proj.weight)
        self.residual_proj.bias.data.zero_()


    def __repr__(self):
        return self.__class__.__name__ + ' ()'

# ===== ===== ===== ===== Residual Convolutional Block with GLU Activations
class ResidualConvBlock(nn.Module):
    """
    A convolutional block with residual connections.

    (At this current moment, we only support stride == 1 due to the complications of padding calculations.)
    """
    def __init__(self, batch_size, in_channels, out_channels, kwidth, stride=1, use_cuda=__CUDA__):
        """
        Construct ResConvBlock.
        """
        # run parent constructor:
        super(ResidualConvBlock, self).__init__()

        # raise error if stride != 1:
        if stride != 1:
            raise Exception("[NANOSEQ] ERR: strides =/= 1 are not supported.")

        # calculate padding: padding is specified as (pleft, pright, ptop, pbottom, pfront, pback)
        if kwidth % 2 == 0:
            #_pad = (int(floor((kwidth-1) / 2)), int(ceil((kwidth-1) / 2)))
            raise Exception("[NANOSEQ] ERR: even-sized kernel widths not supported.")
        else:
            _pad = (int(ceil((kwidth-1) / 2)), int(ceil((kwidth-1) / 2)))

        # store attributes:
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kwidth = kwidth
        self.stride = stride
        self.padding = _pad
        self.use_cuda = use_cuda

        # construct submodules:
        self.conv1d_glu = nn.Conv1d(in_channels, out_channels, kwidth, stride=stride)
        self.conv1d_id = nn.Conv1d(in_channels, out_channels, kwidth, stride=stride)
        self.residual_proj = nn.Linear(in_channels, out_channels)
        self.glu = GLU()

        # move to CUDA if necessary:
        if use_cuda:
            self.conv1d_glu.cuda()
            self.conv1d_id.cuda()
            self.residual_proj.cuda()


    def forward(self, in_seq):
        """
        Forward pass through conv block. Reshape first to (batch, in_depth, seq).

        Args:
        * in_seq: input sequence of shape (seq, batch, in_depth). Pass the sequence through
          convolution blocks and end with a GLU.

        Returns:
        * out_seq: sequence of shape (seq, batch, out_depth).
        """
        # reshape to expected dimensions and apply convolutions:
        in_seq_padded = zero_pad_rank3(in_seq, self.padding, axis=0, use_cuda=self.use_cuda)
        in_seq_reshaped = torch.transpose(torch.transpose(in_seq_padded, 0, 1), 1, 2).contiguous()

        conv_glu = self.conv1d_glu(in_seq_reshaped)
        conv_id = self.conv1d_id(in_seq_reshaped)

        # reshape convolutional outputs:
        conv_glu_reshaped = torch.transpose(torch.transpose(conv_glu, 1, 2), 0, 1).contiguous()
        conv_id_reshaped = torch.transpose(torch.transpose(conv_id, 1, 2), 0, 1).contiguous()

        # apply GLU() and add input sequence:
        glu_out = self.glu(conv_glu_reshaped, conv_id_reshaped)
        in_seq_proj = self.residual_proj(in_seq.view(-1, self.in_channels)).view(
            -1, self.batch_size, self.out_channels)
        return torch.add(glu_out, in_seq_proj)


    def init_params(self):
        """
        Initialize all parameters.
        """
        ### Conv1d-GLU:
        nn_init.xavier_normal(self.conv1d_glu.weight)
        self.conv1d_glu.bias.data.zero_()

        ### Conv1d-Id:
        nn_init.xavier_normal(self.conv1d_id.weight)
        self.conv1d_id.bias.data.zero_()

        ### linear residual projection:
        nn_init.xavier_normal(self.residual_proj.weight)
        self.residual_proj.bias.data.zero_()


    def __repr__(self):
        return self.__class__.__name__ + ' ()'


# ===== ===== ===== ===== Attention Layer
class AttnLayer(nn.Module):
    """
    Attention layer: computes attention over an encoding at each layer of the decoder, using
    dot products.
    """
    def __init__(self, input_dim, encoding_dim, embed_dim, batch_size, use_cuda=False):
        """
        Construct an attention block.

        Args:
        * input_dim: the last dimension of the input sequence. This is usually the hidden sequence from
          the residual convolutional layer.
        * encoding_dim: the last dimension of the encoded sequence.
        * embed_dim: the last dimension from the embedded target sequence; this is the dimension of the latent
          representation.
        * use_cuda: if True, place everything on GPU/CUDA.
        """
        ### parent constructor:
        super(AttnLayer, self).__init__()
        
        ### store parameters as attributes:
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.use_cuda = use_cuda

        ### construct submodules:
        # embeddings to encoding dimension:
        self.in2enc_proj = nn.Linear(input_dim, encoding_dim)
        self.lab2enc_proj = nn.Linear(embed_dim, encoding_dim)
        self.enc2in_proj = nn.Linear(encoding_dim, input_dim)

        ### place on CUDA if requested:
        if use_cuda: [m.cuda() for m in self.modules()]


    def init_params(self):
        """initialize parameters"""
        nn_init.xavier_normal(self.in2enc_proj.weight)
        nn_init.xavier_normal(self.lab2enc_proj.weight)
        nn_init.xavier_normal(self.enc2in_proj.weight)
        self.in2enc_proj.bias.data.zero_()
        self.lab2enc_proj.bias.data.zero_()
        self.enc2in_proj.bias.data.zero_()


    def forward(self, in_seq, enc_seq, prev_target_seq):
        """
        Compute a context vector from input seq, encoded seq, and the sequence of embedded previous targets.
        This effectively performs the same operation on each sequence element of `in_seq`.

        Args:
        * in_seq: FloatTensor variable of shape (len(in_seq), batch_size, input_dim).
        * enc_seq: FloatTensor variable of shape (len(enc_seq), batch_size, encoding_dim).
        * prev_target_seq: embedding of the previous target output; FloatTensor of shape
          (len(in_seq), batch_size, embed_dim).
        Return:
        * context_seq: a sequence of context vector. Of the same type and size as `in_seq`.
        """
        # sanity check:
        assert (len(in_seq) == len(prev_target_seq))


        ### project to same dimension as the encoding:
        in_proj = self.in2enc_proj(in_seq.view(-1, self.input_dim)).view(
            len(in_seq), self.batch_size, self.encoding_dim)
        target_proj = self.lab2enc_proj(prev_target_seq.view(-1, self.embed_dim)).view(
            len(prev_target_seq), self.batch_size, self.encoding_dim)
        d_vals = (in_proj + target_proj) # ~ (len(in_seq), batch, encoding_dim)
        
        ### expand d_val and enc_seq to same (len(in_seq), len(enc_seq), batch_size, encoding_dim) dimensions:
        d_vals_exp = d_vals.unsqueeze(1).expand(d_vals.size(0), enc_seq.size(0), d_vals.size(1), d_vals.size(2))
        enc_seq_exp = enc_seq.unsqueeze(0).expand_as(d_vals_exp)

        ### get attention scores as normalized dot product ~ (len(in_seq), len(enc_seq), batch_size):
        dot_prod_attns_raw = torch.sum(d_vals_exp * enc_seq_exp, 3).squeeze(3)
        dot_prod_norm = torch.sum(dot_prod_attns_raw, 1).expand_as(dot_prod_attns_raw)
        dot_prod_attns = dot_prod_attns_raw / dot_prod_norm

        ### compute context vectors for each timestep; context_seq ~ (len(in_seq), batch_size, encoding_dim)
        context_seq = torch.sum(dot_prod_attns.unsqueeze(3).expand_as(enc_seq_exp) * enc_seq_exp, 1).squeeze(1)

        ### project to correct dimensions so that it can be added to `in_seq` in the decoder, and then return:
        proj_context_seq = self.enc2in_proj(context_seq.view(-1, self.encoding_dim)).view(
            len(in_seq), self.batch_size, self.input_dim)

        return proj_context_seq


# ===== ===== ===== ===== GLU Activation:
class GLU(nn.Module):
    """
    A stateless GLU operation.

    Instead of accepting an even-dimensional block, this accepts two blocks of the same shape
    and performs the GLU operation on them.
    """
    def forward(self, x, y):
        return torch.mul(x,F.sigmoid(y))

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


# ===== ===== ===== ===== Helper functions
def softmax(inp, axis=1):
    """
    SoftMax function with axis argument. Credit to Yuanpu Xie at:
    https://discuss.pytorch.org/t/why-softmax-function-cant-specify-the-dimension-to-operate/2637/2
    """
    # get size of input:
    input_size = inp.size()

    # transpose dimensions:
    trans_input = inp.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    # form 2d (...,axis) tensor:
    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    
    # compute softmax on 2d tensor:
    soft_max_2d = F.softmax(input_2d)
    
    # reshape to original size:
    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size)-1)


def random_pick(batch1, batch2, proba=0.5):
    """
    Flip a weighted coin and return either batch1 or batch2 based on the result.

    Args:
    * batch1: tensor of any type/shape, as long as it is the same as batch2.
    * batch2: tensor of any type/shape, as long as it is the same as batch1.
    * proba: probability of choosing 
    """
    if random.random() < proba:
        return batch1
    else:
        return batch2


def zero_pad_rank3(tsr, padding, axis=1, use_cuda=False):
    """
    Pad a rank-3 tensor with zeros along an axis.

    Args:
    * tsr: of shape [ax1, ax2, ax3].
    * padding: python integer tuple (pad_start, pad_end) that indicates the number of zeros to add to the front
      and back of some axis of `tsr`.
    * axis: axis along which to pad.
    * use_cuda: whether to allocate tensors to CUDA.
    """
    # if no padding, return:
    if padding[0] < 1 and padding[1] < 1:
        return tsr

    # figure out padding shape:
    tsr_shape = tsr.size()
    zero_pad_start_shape = list(tsr_shape)
    zero_pad_stop_shape = list(tsr_shape)
    zero_pad_start_shape[axis] = padding[0]
    zero_pad_stop_shape[axis] = padding[1]
    
    if use_cuda:
        if padding[0] > 0:
            zero_pad_start = torch.autograd.Variable(torch.zeros(zero_pad_start_shape).cuda())
        if padding[1] > 0:
            zero_pad_stop = torch.autograd.Variable(torch.zeros(zero_pad_stop_shape).cuda())
    else:
        if padding[0] > 0:
            zero_pad_start = torch.autograd.Variable(torch.zeros(zero_pad_start_shape))
        if padding[1] > 0:
            zero_pad_stop = torch.autograd.Variable(torch.zeros(zero_pad_stop_shape))

    # concat zeros along axis:
    if padding[0] > 0 and padding[1] > 0:
        _pad_sequence = (zero_pad_start, tsr, zero_pad_stop)
    if padding[0] < 0 and padding[1] > 0:
        _pad_sequence = (tsr, zero_pad_stop)
    if padding[0] > 0 and padding[1] < 0:
        _pad_sequence = (zero_pad_start, tsr)

    return torch.cat(_pad_sequence, dim=axis)


def sequence_norm(tsr, axis=1, eps=1e-8):
    """
    Normalize a 3D tensor across a sequence-dimension.

    (N.B.: event sequences have START/STOP/PAD bits, but these can be averaged without affecting their values
    as they are one-hot indicators.)
    """
    mu = torch.mean(tsr, axis)
    sigma = torch.std(tsr, axis)
    sigma = torch.max(sigma, eps * torch.ones(sigma.size()))

    mu_stack = torch.stack([mu] * tsr.size(axis), axis)
    sigma_stack = torch.stack([sigma] * tsr.size(axis), axis)

    return (tsr - mu_stack) / sigma_stack
