"""
langmod_lstm.py

Core class for defining the multi-layer LSTM-based Language Model. Consists of an embedding layer
mapping hot-encodings to the embedding space, then num_layers stacked LSTM layers, then a final
softmax layer, for predictions.
"""
from tensorflow.models.rnn import rnn
import tensorflow as tf

# Get parameters
FLAGS = tf.flags.FLAGS


class LangmodLSTM:
    def __init__(self, batch_size, window_size):
        """
        Initialize a multi-layer LSTM-based language model.
        """
        # Collect model parameters
        self.batch_size, self.window_size = batch_size, window_size
        self.learning_rate, self.lr_decay = FLAGS.learning_rate, FLAGS.lr_decay
        self.embedding_size = self.hidden_size = FLAGS.embedding_size
        self.max_grad_norm = FLAGS.max_grad_norm
        self.vocab_size, self.lstm_layers = FLAGS.vocab_size, FLAGS.lstm_layers

        # Setup placeholders
        self.X = tf.placeholder('int64', [None, self.window_size])
        self.Y = tf.placeholder('int64', [None, self.window_size])

        # Setup learning_rate placeholder
        self.lr = tf.Variable(0.0, trainable=False)

        # Setup initial, final LSTM states
        self.initial_state = self.final_state = None

        # Build the model, return the logits
        self.logits = self.inference()

        # Compute the loss
        self.loss_val = self.loss()

        # Set up the training operation.
        self.train_op = self.train()

    def assign_lr(self, session, value):
        """
        Assign learning rate to a different value --> Placeholder doesn't work because
        a scalar, not a tensor.

        :param session: Tensorflow session.
        :param value: New Learning Rate (for decay).
        """
        session.run(tf.assign(self.lr, value))

    def inference(self):
        """
        Build the neural network graph, using placeholders for the input (to be fed during
        training).

        :return: Logits tensor, after being passed through the Neural Network Graph.
        """
        # Build embedding
        embed_matrix = init_weight([self.vocab_size, self.embedding_size], 'Embedding_Matrix')
        embedding = tf.nn.embedding_lookup(embed_matrix, self.X)

        # Instantiate a single LSTM Layer
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=0.0)

        # Stack LSTM cells
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.lstm_layers)

        # Setup LSTM initial state
        self.initial_state = multi_cell.zero_state(self.batch_size, tf.float32)

        # Feed through LSTM
        outputs, state = rnn.rnn(multi_cell, embedding, initial_state=self.initial_state)

        # Compute final state
        self.final_state = state

        # Build Softmax Layer --> Softmax(Wx + b)
        softmax_weight = init_weight([self.hidden_size, self.vocab_size], 'Softmax_Weight')
        softmax_bias = init_weight([self.vocab_size], 'Softmax_Bias')

        # Compute logits
        logits = tf.matmul(outputs, softmax_weight) + softmax_bias
        return logits

    def loss(self):
        """
        Build the computation graph for calculating the cross entropy loss.
        """
        loss = tf.nn.seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(self.Y, [-1])],
                                                      tf.ones([self.batch_size, self.window_size]))
        cost = tf.reduce_sum(loss) / float(self.batch_size)
        tf.scalar_summary('Loss', cost)  # Add a summary to track Loss over time
        return cost

    def train(self):
        """
        Build the computation graph for computing gradients and doing backprop. Rather than have
        the optimizer just minimize the cost, we directly compute gradients (after clipping norm),
        and have it apply those via backprop.
        """
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_val, tvars), self.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        return optimizer.apply_gradients(zip(grads, tvars))


def init_weight(shape, name):
    """
    Initialize a Tensor corresponding to a weight matrix with the given shape and name.

    :param shape: Shape of the weight tensor.
    :param name: Name of the weight tensor in the computation graph.
    :return: Tensor object with given shape and name, initialized from a standard normal.
    """
    return tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)