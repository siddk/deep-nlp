# Deep - NLP #

This repository contains Tensorflow implementations of various deep learning models, with a focus
on problems in Natural Language Processing. Each individual subdirectory is self-contained, addressing
one specific model. 

## Models ##
The following models are implemented:

+ **mnist_nn/**: A simple one-layer classifier implemented as part of the Tensorflow Tutorial for the
                 MNIST Handwriting Classification task. Very simple model, contained in a single 
                 python script, just to show off the Tensorflow basics.
                 
+ **langmod_nn/**: A three-layer Feed-Forward Bigram Language model that tries to predict the next 
                   word in the corpus given the current word. 
                   
+ **cifar_cnn/**: A multi-layer Convolutional Neural Network that follows the AlexNet convention for
                  the CIFAR-10 image classification task. Given an image, classify it as an 
                  airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.
                  
## Project Setup ##
There are a lot of ways to set up and train Tensorflow models, which can be very confusing. With
the exception of the simple MNIST NN from the Tensorflow Tutorial, each of the above model 
subdirectories is broken up in the following way:

**train.py**: This file is the core file for training a given model (think of this as the main 
              script for the entire directory). This script loads the data, performs any 
              preprocessing to format the data for training and testing, and then builds and trains
              the model. Usually contains code to output training details to stdout, as well as 
              code to save/serialize the model parameters periodically.
              
**preprocessor/**: Subdirectory that contains any data processing code (i.e. code to read raw data
                   like text or images, and convert it to numerical form for training/testing).
                   
**model/**:
  
  - **model.py**: Class definition for the model's neural network. Tensorflow at its core is a 
                  system for building symbolic computational graphs, and everything in Tensorflow
                  is either expressed as a raw Tensor, or a Tensor operation. Because of this,
                  building a model consists of building different graphs and operations to handle
                  the inference of the model, how to evaluate the loss/cost, and how to perform
                  training (usually via backpropagation). 
                  Because of this, each class definition consists of the following three functions:
    
    + *inference*: This is the crux of any neural network. This function is responsible for building
                   all the layers of the network, from the input, all the way to the final layer,
                   just before the loss is calculated.
    
    + *loss*: Using the output from the *inference* function, this function evaluates the loss used
              for training the model. For example, the loss function might take in the *logits* from
              the softmax layer of a classification model (say like in the MNIST model), and calculate
              the cross-entropy loss with the true labels of the input data.
              
    + *train*: The train function builds the training operation, given the cost calculated in the
               *loss* function. This function computes the gradients, and sets up the optimizer
               (i.e. SGD, Adam, Adagrad, etc.). Any learning rate decay is also performed during
               this step.
               
**data/**: A data subdirectory, for storing raw data.

**log/**: A log directory, consisting of two parts - summaries, and checkpoints. Each of the above
          Tensorflow models have code to store Tensorboard-formatted Summary files to track things
          like loss over time, accuracy, gradients, etc, and these logs are stored in `logs/summaries`.
          The models also have code to save and serialize all the parameters during training, allowing
          for easy restoration of interrupted training, or for just loading fully trained models.
 