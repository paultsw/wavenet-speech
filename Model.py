import torch.nn as nn
from torch.autograd.Variable
from modules.wavenet import WaveNet
from modules.classifier import WaveNetClassifier as CTCNet
from modules.raw_ctcnet import RawCTCNet

class Model(object):
    """
    Wrapper for all model architectures in this project.
    """
    def __init__(self, model_type, cfg):
        """
        Generate a model. Currently supports two types of model: WaveNetCTC and RawCTCNet.
        WaveNetCTC works on quantized (one-hot-encoded) signal data whereas RawCTCNet works
        on a one-dimensional signal.
        """
        assert model_type in ['wavenet-ctc', 'raw-ctcnet']
        self.model_type = model_type

        # unpack config and construct wavenet-ctc model:
        if model_type == 'wavenet-ctc':
            num_levels = 256
            wavenet_kwidth = 2
            num_ctc_dilation_blocks = 5
            wavenet_dils = [1,2,4,8,16,32,64] * num_dilation_blocks
            self.model_base = WaveNet(num_levels, wavenet_kwidth,
                                      [(num_levels, num_levels, wavenet_kwidth, d) for d in wavenet_dils],
                                      num_levels, softmax=False)
            num_labels = 5
            num_ctc_dilation_blocks = 5
            ctc_dils = [1,2,4,8,16,32,64] * num_ctc_dilation_blocks
            out_dim = 512
            downsample_rate = 1
            self.model_ctc = CTCNet(num_levels, num_labels,
                                    [(num_levels, num_levels, classifier_kwidth, d) for d in classifier_dils],
                                    out_dim, pool_kernel_size=downsample_rate,
                                    input_kernel_size=2, input_dilation=1,
                                    softmax=False)

        # unpack config and construct rawctcnet model:
        if model_type == 'raw-ctcnet':
            nfeats = 2048
            feature_kwidth = 3
            num_labels = 5
            num_dilation_blocks = 20
            dils = [1,2,4,8,16,32,64] * num_dilation_blocks
            out_dim = 512
            use_causal = False
            layers = [() for d in dils]
            self.model_base = nn.BatchNorm1d(1)
            self.model_ctc = RawCTCNet(nfeats, feature_kwidth, num_labels, layers,
                                       out_dim, input_kernel_size=2, input_dilation=1,
                                       softmax=False, causal=use_causal)

    def predict(self, signal):
        """Generated a predicted transcript from the input signal."""
        model_base_out = self.model_base(signal)
        transcript_out = self.model_ctc(model_base_out)
        return (model_base_out, transcript_out)

    def get_parameters(self):
        """Return either a list of dictionaries of parameters, as expected by torch.optim.Optimizer."""
        return [{'params': self.model_base.parameters()}, {'params': self.model_ctc.parameters()}]


    def restore(self, restore_path_base, self.restore_path_ctc):
        """Restore model weights."""
        self.model_base.load_state_dict(torch.load(restore_path_base))
        self.model_ctc.load_state_dict(torch.load(restore_path_ctc))


    def save(self, save_path_base, save_path_ctc):
        """Save model weights."""
        torch.save(self.model_base.state_dict(), save_path_base)
        torch.save(self.model_ctc.state_dict(), save_path_ctc)
