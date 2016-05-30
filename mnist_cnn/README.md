# MNIST CNN #
Builds a multiple layer convolutional neural network for the MNIST Handwriting Digit Classification
task. Given an image of a handwritten number, classify which digit is depicted. 

## Model Setup ##

+ **Input**: A 28 x 28 Pixel Image. Some preprocessing steps are done, turning each image into a
             4D Tensor of shape (Batch Size, Image Length, Image Width, Depth), where the Depth 
             (normally an integer between 0 and 255) is scaled down into the range -0.5 to 0.5.
  
+ **Output**: Vector of size 10, corresponding to the probability that the given image belongs to
              each of the 10 different digit classes (0 - 9). Because this is a vector of 
              probabilities, they are referred to as "logits" in the code. Note that the final 
              softmax layer doesn't actually occur in the training, it is done implicitly during 
              the calculation of the loss.

+ **Layers**: This model follows a fairly standard CNN Convention. Two Convolution + Max Pooling 
              Layers, then two feed-forward layers, with the final being the softmax over the 10
              digit classes. A more specific breakdown is as follows (names follow code convention):
            
  1) **Conv1**: Convolution Layer with ReLU Activation.
  
  2) **Pool1**: Max Pooling Layer.
  
  3) **Conv2**: Convolution Layer with ReLU Activation.
  
  4) **Pool2**: Max Pooling Layer.
  
  5) **Reshape**: Not really a layer, but the Pool2 output is reshaped into a 2D Tensor.
  
  6) **Fc1**: Fully Connected Layer with ReLU Activation, consisting of 512 units.
  
  7) **Fc2**: Second Fully Connected Layer, with Softmax Activation. Outputs the size-10 logits 
              vector.
               
+ **Loss**: The loss function used is the sum of the standard cross-entropy loss between the 
            predicted probabilities and the true labels, and the L2 Norm of the weights of the two
            feed-forward layers (for better regularization).
            
+ **Optimizer**: The Momentum Optimizer is used with an exponentially decaying learning rate, with
                 base learning rate .01, and decay rate 0.95.

## Directory Setup ##

The core file for training the model is `train.py` in the root of the current directory. Running 
`python train.py` loads the train and test data from the `data` directory, parses it using the code
in the `preprocessor` subdirectory, and then builds the model computation graph using the MnistCNN 
class code in the `model` subdirectory. A specific breakdown is as follows:

+ **preprocessor/**: This directory contains a single file, `loader.py` that has a function that
downloads the MNIST corpus from Yann LeCun's website, and performs all the necessary preprocessing,
eventually returning the training, validation, and test data.

+ **model/**: This directory contains the class definition for the `MnistCNN` class. I use the 
convention of initializing the model with all the model-specific hyperparameters 
(layer size, learning rate, etc.), then building each of the computation graphs for doing inference, 
calculating loss, then performing training. Each of these subgraphs are accessible at training time 
by doing a field access on a specific instance of the model (see `train.py` for an example).

+ **log/**: The log directory consists of the following two subdirectories:
    
    1) **checkpoints/**: This consists of the saved checkpoints of the model's variables. During the
                        training process, the entire model is serialized and stored here at the end
                        of each training epoch.
    
    2) **summaries/**: This consists of the summary logs for the running of the model. By pointing
                      Tensorboard to this directory, you can visualize the computation graph, as well
                      as track the loss function over time.
                      
## Results ##

During training, a batch size of 64 is used, with 55,000 total training examples, 5,000 total 
validation examples, and 10,000 test examples. The model is trained for 10 epochs, each of which
takes a few minutes on a CPU (much faster on a GPU). The model is evaluated on the validation data
every 100 batches.

Training loss (evaluated at 100 batch intervals):

![img](https://github.com/siddk/deep-nlp/blob/master/mnist_cnn/data/loss.png)

Training accuracy (evaluated at 100 batch intervals):

![img](https://github.com/siddk/deep-nlp/blob/master/mnist_cnn/data/accuracy.png)

Learning Rate over time (also evaluated at 100 batch intervals):

![img](https://github.com/siddk/deep-nlp/blob/master/mnist_cnn/data/learning_rate.png)

This model has a **Test Accuracy of 99.1%**.