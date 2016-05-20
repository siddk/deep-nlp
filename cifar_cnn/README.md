# CIFAR-10 CNN #
Builds a multiple layer Convolutional Neural Network for the CIFAR-10 Task of categorizing images.
Given an image, classify it as one of ten different categories: airplane, automobile, bird, cat, 
deer, dog, frog, horse, ship, and truck.

## Model Setup ##
+ **Input**: 28 x 28 Pixel Image. At training time, a series of distortions are performed, to help
             with model regularization. At evaluation time, the regular images are used.
             
+ **Output**: Vector of size 10, corresponding to the probability that the given image belongs to
              each of the 10 CIFAR classes. Because this is a vector of probabilities, they are
              referred to as "logits" in the code.