# Feed - Forward Bigram Model #
Builds a three-layer neural network consisting of an Embedding Layer, a Hidden Layer, and a final
Softmax layer where the goal is as follows: Given a word in a corpus, attempt to predict the next 
word.

## Model Setup ##
+ **Input** A word in a corpus. Because the vocabulary size can get very large, we have limited the
            vocabulary to the top 5000 words in the corpus, and the rest of the words are replaced
            with the UNK symbol. Each sentence in the corpus is also doubly-padded with a Stop symbol.
            To pass the word into the model, it is encoded one-hot in a vector the size of the vocabulary
            (dimension 5000).

+ **Output** The following word in the corpus, also encoded one-hot in a vector the size of the vocabulary.

+ **Layers**: The model consists of the following three layers:

  1) **Embedding Layer**: Each word corresponds to a unique embedding vector, a representation of the
                      word in some embedding space. Here, the embeddings all have dimension 50. We 
                      find the embeddings for a given word by doing a matrix multiply (essentially
                      a table lookup) with an Embedding Matrix that is trained during regular backprop.
  
  2) **Hidden Layer**: Fully-Connected Feed-Forward Layer with Hidden Layer size 100, and ReLU Activation.
  
  3) **Softmax Layer**: A Fully-Connected Feed-Forward Layer with Layer size equal to the Vocab Size,
                    where each element of the output vector (logits) corresponds to the probability 
                    of that word in the vocabulary being the next word.
                    
+ **Loss**: The normal Cross-Entropy loss between the logits and the true labels as the model's
            cost.
            
+ **Optimizer**: A normal SGD Optimizer with learning rate .05.

## Directory Setup ##

The core file for training the model is `train.py` in the root of the current directory. Running 
`python train.py` loads the train and test data from the `data` directory, parses it using the code
in the `preprocessor` subdirectory, and then builds the model computation graph using the Langmod 
class code in the `model` subdirectory. A specific breakdown is as follows:

+ **preprocessor/**: This directory contains a single file, `reader.py` that has a function that
reads in the raw corpus, and converts it to the input and output bigram pairs for training.

+ **model/**: This directory contains the class definition for the `Langmod` class. I use the convention
of initializing the model with all the model-specific hyperparameters (layer size, learning rate, etc.),
then building each of the computation graphs for doing inference, calculating loss, then performing
training. Each of these subgraphs are accessible at training time by doing a field access on a 
specific instance of the model (see `train.py` for an example).

+ **log/**: The log directory consists of the following two subdirectories:
    
    1) **checkpoints/**: This consists of the saved checkpoints of the model's variables. During the
                        training process, the entire model is serialized and stored here at the end
                        of each training epoch.
    
    2) **summaries/**: This consists of the summary logs for the running of the model. By pointing
                      Tensorboard to this directory, you can visualize the computation graph, as well
                      as track the loss function over time.
                      
## Results ##

Each epoch (around 480,000 examples) takes about 10 minutes to train on CPU. The following is a 
graph of training and test loss over time:

Training loss (evaluated at each batch --> leads to high variance):

![img](https://github.com/siddk/deep-nlp/blob/master/langmod_nn/log/train.png)

Test loss (evaluated after each epoch):

![img](https://github.com/siddk/deep-nlp/blob/master/langmod_nn/log/test.png)

Further evaluation (log-likelihood of corpus, etc.) TBA.