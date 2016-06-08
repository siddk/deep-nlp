"""
langmod.py

Core class for defining the Simple Three-Layer Feed-Forward Language Model. Consists of an embedding
layer mapping one-hot word encodings to high dimensional space, then a ReLU layer, then a final
softmax layer.
"""
import tensorflow as tf

# Get current parameters
FLAGS = tf.app.flags.FLAGS


class Langmod:
    def __init__(self):
        """
        Initialize a Langmod Neural Network.
        """
        self.epochs, self.batch_size = FLAGS.epochs, FLAGS.batch_size
        self.learning_rate, self.vocab_size = FLAGS.learning_rate, FLAGS.vocab_size
        self.embedding_size, self.hidden_size = FLAGS.embedding_size, FLAGS.hidden_size

        # Setup placeholders
        self.X = tf.placeholder('int64', [None])  # Input Shape: (batch) -> Index of word in vocab
        self.Y = tf.placeholder('int64', [None])  # Label Shape: (batch) -> Index of word in vocab

        # Build the model, return the logits.
        self.logits = self.inference()

        # Build the loss computation graph, using the logits.
        self.loss_val = self.loss()

        # Calculate the log likelihood of the data, using the logits.
        self.log_lik = self.log_likelihood()

        # Set up the training operation.
        self.train_op = self.train()

    def inference(self):
        """
        Build the neural network graph, using placeholders for the input (to be fed during
        training).

        :return: Logits tensor, after being passed through the Neural Network Graph.
        """
        one_hot_x = tf.one_hot(self.X, self.vocab_size, 1.0, 0.0)
        embed_matrix = init_weight([self.vocab_size, self.embedding_size], 'Embedding_Matrix')
        embedding = tf.matmul(one_hot_x, embed_matrix)        # Shape is now (batch, embed_sz)

        hidden_matrix = init_weight([self.embedding_size, self.hidden_size], 'Hidden_Matrix')
        hidden = tf.matmul(embedding, hidden_matrix)
        hidden_relu = tf.nn.relu(hidden)                      # Shape is now (batch, hidden_sz)

        softmax_matrix = init_weight([self.hidden_size, self.vocab_size], 'Softmax_Matrix')
        softmax = tf.matmul(hidden_relu, softmax_matrix)      # Shape is now (batch, vocab_sz)

        return softmax                                        # Actual softmax done during loss eval

    def loss(self):
        """
        Build the computation graph for calculating the cross entropy loss.
        """
        one_hot_y = tf.one_hot(self.Y, self.vocab_size, 1.0, 0.0)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, one_hot_y, name='Loss'))
        tf.scalar_summary('Loss', cost)  # Add a summary to track Loss over time

        return cost

    def log_likelihood(self):
        """
        Build the computation graph for calculating the log likelihood of the data.
        """
        softmax = tf.nn.softmax(self.logits)
        one_hot_y = tf.one_hot(self.Y, self.vocab_size, 1.0, 0.0)
        probabilities = tf.reduce_max(tf.mul(softmax, one_hot_y), reduction_indices=[1])
        log_lik = tf.reduce_sum(tf.log(probabilities))
        tf.scalar_summary('Log Likelihood', log_lik)

        return log_lik

    def train(self):
        """
        Build the computation graph for computing gradients and doing backprop.
        """
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss_val)


def init_weight(shape, name):
    """
    Initialize a Tensor corresponding to a weight matrix with the given shape and name.

    :param shape: Shape of the weight tensor.
    :param name: Name of the weight tensor in the computation graph.
    :return: Tensor object with given shape and name, initialized from a standard normal.
    """
    return tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)