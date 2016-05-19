# MNIST NN #
Simple MNIST Handwriting Classification Problem. Given a 28 x 28 pixel image, pick which digit class
it belongs to.

## Model Setup ##
    
+ **Input**: 28 x 28 Pixel Image, flattened into a single vector of size 784. The input "x" is then
         a matrix of size (Batch-Size, 784).
    
+ **Output**: Vector of size 10, corresponding to the probability that the given image belongs to
          each of the 10 digit classes (0, 1, 2, ... 9). The output "y" is then a matrix of 
          size (Batch-Size, 10).
              
+ **Layers**: The model only has one fully-connected feed-forward layer, with a bias vector. This
          layer is followed by a Softmax Layer.
  
  1) **Feed-Forward Layer**: This is represented by a weight matrix "W" of size (784, 10). 
                           The bias vector is of size (10). The transformation is of the form 
                           (xW + b), a simple matrix multiplication between the input and the 
                           weight matrix, and the addition of the bias.
         
  2) **Softmax Layer**: Normal softmax over the output of size (10). Represents probability that
                    the current example image belongs to each of the 10 digit classes.
    
+ **Loss**: We use cross-entropy loss. The true predictions are encoded as one-hot versions of the 
        true classes (vectors of size 10).
            
+ **Optimizer**: Simple SGD Optimizer with learning rate 0.5.