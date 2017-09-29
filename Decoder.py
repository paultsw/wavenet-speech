from modules.sequence_decoders import argmax_decode, labels2strings, BeamSearchDecoder

_DEFAULT_BEAM_MAP_ = { '<pad>': 0, '<s>': 5, '</s>': 6 } # mapping of special symbols required for beam decoder
class Decoder(object):
    """Abstract wrapper class that decodes a sequence of logits."""
    def __init__(self, decoder, batch_size, num_labels, mapping_dict=_DEFAULT_BEAM_MAP_, beam_width=None, cuda=False):
        """Construct everything you need for a decoder."""
        assert decoder in ['argmax', 'beam']
        self.decoder_type = decoder
        if decoder == 'beam':
            self.beam_decoder = BeamSearchDecoder(batch_size, num_labels, mapping_dict=mapping_dict,
                                                  beam_width=beam_width, cuda=cuda)


    def decode(self, logits):
        """
        Decode a sequence of logits and return decoded strings.

        `logits` is assumed to be of shape (batch, num_labels, sequence_length).
        """
        lookup_dict = {0: '', 1: 'A', 2: 'G', 3: 'C', 4: 'T'}
        if self.decoder_type == 'argmax':
            decoded = argmax_decode(logits.permute(0,2,1).contiguous())
            parsed_decoded = labels2strings(decoded, lookup=lookup_dict)
            probas = None
        else:
            probas, decoded = self.beam_decoder.decode(logits)
            parsed_decoded = [labels2strings(b, lookup=lookup_dict) for b in decoded]

        return (probas, parsed_decoded)
