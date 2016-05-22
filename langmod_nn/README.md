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
  