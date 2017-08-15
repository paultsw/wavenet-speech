WaveNet-Speech
==============

Implementation of Google Deepmind's WaveNet for speech recognition, written in PyTorch.

See section 3.4 of the paper [WaveNet: A Generative Model for Raw Audio (PDF)](`https://arxiv.org/pdf/1609.03499.pdf) for more details on how the speech recognition setup differs from the standard WaveNet architecture.

The fundamental difference between "vanilla" WaveNet and the speech recognition WaveNet is a classifier network stacked on top of the typical WaveNet as well as a training loop that optimizes a loss given by a sum of two loss functions.

The first loss function is the typical (log-likelihood) loss of the WaveNet component (say, `Loss1 := NLLLoss(WaveNet(input_sequence))`. The second loss function is the CTC loss on the output of the classifier network (say, `Loss2 := CTCLoss(Classifier(WaveNet(input_sequence)))`. We then minimize the sum `Loss = Loss1 + Loss2` during training.

WaveNet Architecture
--------------------
`(TBD: describe basic WaveNet architecture)`

Classifier Network
------------------
The classifier network is composed of a mean-pooling layer applied to the output of the WaveNet with non-causal (i.e., standard) Conv1d layers stacked on top of it.