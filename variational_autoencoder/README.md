# MNIST Variational Autoencoder #
Variational Autoencoder for the MNIST Handwritten Digits dataset. As a Variational Autoencoder,
the goal of this model is to simulate a generative model. 

In this case, the model takes an input MNIST digit and maps (encodes) it to some latent space (in 
the code, this space is referred to as z). It then takes this latent space representation and 
maps (decodes) it to the data space, trying to reconstruct the original image. A really nice 
tutorial of Variational Autoencoders can be found [here](https://arxiv.org/pdf/1606.05908v1.pdf).
A large portion of the setup for this model was derived from 
[here](https://jmetzen.github.io/2015-11-27/vae.html). 

## Results ##

Here are the results of the VAE trying to reconstruct an image given a seed from the MNIST test
set, in 20-dimensional latent space:

<img src="https://github.com/siddk/deep-nlp/blob/master/variational_autoencoder/results/reconstructed.png" width="500">

