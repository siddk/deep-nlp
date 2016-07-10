"""
langmod_rnn.py

Core model definition file for the LSTM Language Model. Defines a three layer stacked LSTM
language model, with an embedding layer, and a final softmax layer.
"""
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


class LangmodLSTM:
    def __init__(self):
        """
        Initialize a LangmodLSTM RNN Language Model, with the necessary parameters.
        """
        # Setup Placeholders
        self.X = tf.placeholder(tf.int64, [None, None])  # Shape [bsz, window_size]
        self.Y = tf.placeholder(tf.int64, [None, None])  # Time-shifted X

        # Instantiate Weights
        self.instantiate_weights()

        # Build the Model, return the Logits
        self.logits = self.inference()

        # Build the loss computation graph, using the Logits
        self.loss_val = self.loss()

        # Set up Training Graph
        self.lr = tf.Variable(0.0, trainable=False)
        self.train_op = self.train()

    def instantiate_weights(self):
        """
        Initialize all the Trainable Variables for use in the inference pipeline.
        """
        self.embedding = init_weight([FLAGS.vocab_size, FLAGS.embedding_size], "Embedding")
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_size, forget_bias=0.0)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * FLAGS.num_layers)
        self.initial_state = self.cell.zero_state(FLAGS.batch_size, tf.float32)
        self.softmax_weight = init_weight([FLAGS.hidden_size, FLAGS.vocab_size], "SoftmaxWeight")
        self.softmax_bias = init_bias(FLAGS.vocab_size, 0.1, "SoftmaxBias")

    def inference(self):
        """
        Build the core computation graph, from the inputs to the logits.
        """
        # Input -> Embeddings [batch, window, embedding]
        inputs = tf.nn.embedding_lookup(self.embedding, self.X)

        # Transform Input Tensor into List required by tf.rnn
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, FLAGS.window_size, inputs)]

        # Feed to Stacked LSTM
        outputs, state = tf.nn.rnn(self.cell, inputs, initial_state=self.initial_state)
        self.final_state = state

        # Reshape Outputs from List into Single Tensor
        output = tf.reshape(tf.concat(1, outputs), [-1, FLAGS.hidden_size])

        # Build and return final logits (w/o softmax)
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        return logits

    def loss(self):
        """
        Build the loss computation graph, using the Logits and the true labels.
        """
        chunk_size = FLAGS.batch_size * FLAGS.window_size
        loss = tf.nn.seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(self.Y, [-1])],
                                                      [tf.ones([chunk_size])])
        return tf.reduce_sum(loss) / FLAGS.batch_size

    def train(self):
        """
        Build the training operation, for backpropagation.
        """
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_val, tvars), FLAGS.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        return optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))


def init_weight(shape, name):
    """
    Initialize a Tensor corresponding to a weight matrix with the given shape and name.

    :param shape: Shape of the weight tensor.
    :param name: Name of the weight tensor in the computation graph.
    :return: Tensor object with given shape and name, initialized from a standard normal.
    """
    return tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)


def init_bias(shape, value, name):
    """
    Initialize a Tensor corresponding to a bias vector with the given shape and name.

    :param shape: Shape of the bias vector (as an int, not a list).
    :param value: Value to initialize bias to.
    :param name: Name of the bias vector in the computation graph.
    :return: Tensor (Vector) object with given shape and name, initialized with given bias.
    """
    return tf.Variable(tf.constant(value, shape=[shape]), name=name)