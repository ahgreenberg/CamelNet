# CamelNet

CamelNet is a translationally invariant denoising convolutional autoencoder. The compression ratio is >90% (from a 256 dimensional input vector to a 24 dimensional encoded vector)

The encoder is seven layers deep - not including two max-pooling layers - consisting of two convolutional layers and four fully-connected linear layers. 
The decoder is four layers deep, consisting of three transpose convolutional layers and a final fully-connected linear layer. 
The activation functions throughout are leaky ReLUs

CamelNet was trained with a novel loss function to enable fast convergence on a translationally invariant encoding. 
This loss function includes a regularization term that penalizes variance in encoded vectors across an ensemble of input vectors that differ only by noise and translation
