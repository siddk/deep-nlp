# CIFAR-10 CNN #
Builds a multiple layer Convolutional Neural Network for the CIFAR-10 Task of categorizing images.
Given an image, classify it as one of ten different categories: airplane, automobile, bird, cat, 
deer, dog, frog, horse, ship, and truck.

Note: Currently no results shown, as the model takes several hours to train on GPU.

## Model Setup ##
+ **Input**: 28 x 28 Pixel Image. At training time, a series of distortions are performed, to help
             with model regularization. At evaluation time, the regular images are used.
             
+ **Output**: Vector of size 10, corresponding to the probability that the given image belongs to
              each of the 10 CIFAR classes. Because this is a vector of probabilities, they are
              referred to as "logits" in the code. Note that the final softmax layer doesn't 
              actually occur in the training, it is done implicitly during the call to the loss
              function.
              
+ **Layer**: This model follows the AlexNet CNN convention. See AlexNet paper for detailed 
             descriptions of layers. The layers are broken up as follows:
  
  1) **Conv1**: Convolution Layer with ReLU Activation.
  
  2) **Pool1**: Max Pooling Layer.
  
  3) **Norm1**: Localized Response Normalization Layer.
  
  4) **Conv2**: Convolution Layer with ReLU Activation.
  
  5) **Norm2**: Localized Response Normalization Layer.
  
  6) **Pool2**: Max Pooling Layer.
  
  7) **Local1**: Fully Connected Feed-Forward Layer with ReLU Activation.
  
  8) **Local2**: Another Fully Connected Feed-Forward Layer with ReLU Activation.
  
  9) **Softmax Linear Transformation**: Simple Linear Transformation with Bias, then a Softmax. Or,
              in other words, if the output of the previous layer is "x", then this layer returns
              Softmax(Wx + b), where "W" and "b" are weight matrices.

+ **Loss**: We the sum of the cross-entropy loss between the logits and the true predictions and the
            L2 Norms of the weight matrices at each layer (to help with regularization). 
            
+ **Optimizer**: Here we use a special version of the SGD optimizer with a Learning Rate that decays
                 exponentially over time.
  
  