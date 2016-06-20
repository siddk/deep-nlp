"""
mnist_cnn.py

Core class for defining the multi-layer MNIST Convolutional Neural Network. Consists of 2 sets
convolutional layers (convolution and pooling), and two sets of feed-forward fully connected layers.
"""
from __future__ import division
import tensorflow as tf

# Get Tensorflow Parameters
FLAGS = tf.flags.FLAGS


class MnistCNN:
    def __init__(self):
        """
        Initialize a MnistCNN model with the necessary hyperparameters.
        """
        self.batch_size = FLAGS.batch_size
        self.image_size, self.channels = FLAGS.image_size, FLAGS.num_channels
        self.hidden, self.num_labels = FLAGS.hidden_size, FLAGS.num_labels
        self.l2_discount = FLAGS.l2_discount
        self.lr, self.lr_decay = FLAGS.learning_rate, FLAGS.lr_decay
        self.train_size = FLAGS.train_size

        # Initialize Placeholders
        input_shape = [self.batch_size, self.image_size, self.image_size, self.channels]
        self.X = tf.placeholder(tf.float32, shape=input_shape)
        self.Y = tf.placeholder(tf.int64, shape=[self.batch_size])
        self.eval_X = tf.placeholder(tf.float32, shape=input_shape)

        # Instantiate trainable weights (do separately so validation/training can share)
        self.instantiate_weights()

        # Build the model, return the logits and the fully-connected weights (for regularization)
        self.logits = self.inference(True)

        # Build prediction ops using the logits for both training and eval.
        self.train_prediction = tf.nn.softmax(self.logits)
        self.eval_prediction = tf.nn.softmax(self.inference(False))

        # Build Accuracy ops using the predictions, add summaries
        self.train_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.train_prediction, 1), self.Y), "float"))
        tf.scalar_summary('Training_Accuracy', self.train_acc)

        # Build the loss computation graph, using the logits
        self.loss_val = self.loss()

        # Set up the training_operation
        self.learning_rate, self.train_op = self.train()

    def instantiate_weights(self):
        """
        Initialize all the trainable weight tensors for training.
        """
        # 1st Convolution Layer Weights and Bias
        self.conv1_w = init_weight([5, 5, self.channels, 32], 'Conv1_Weight')  # 5x5 filter, 32 deep
        self.conv1_b = init_bias(32, 0.0, 'Conv1_Bias')

        # 2nd Convolution Layer Weights and Bias
        self.conv2_w = init_weight([5, 5, 32, 64], 'Conv2_Weight')  # 5x5 filter, depth 64
        self.conv2_b = init_bias(64, 0.1, 'Conv2_Bias')

        # Fully Connected Layer 1 -> ReLU Activation
        self.fc1_w = init_weight([self.image_size // 4 * self.image_size // 4 * 64, self.hidden],
                            'FC1_Weight')
        self.fc1_b = init_bias(self.hidden, 0.1, 'FC1_Bias')

        # Fully Connected Layer 2 -> for softmax (actual softmax performed in loss function)
        self.fc2_w = init_weight([self.hidden, self.num_labels], 'FC2_Weight')
        self.fc2_b = init_bias(self.num_labels, 0.1, 'FC2_Bias')

    def inference(self, train=False):
        """
        Build the core of the model, initialize all convolutional and feed-forward layers, with the
        respective weights, and add dropout if necessary.

        :param train: Boolean if training or eval, necessary for including dropout.
        :return: Tuple of resulting logits, and the feed-forward trainable weights for L2 Loss.
        """
        # 2D Convolution Layer, then Bias + ReLU, then Pooling Layer (add conditional train/eval)
        if train:
            conv1 = tf.nn.conv2d(self.X, self.conv1_w, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = tf.nn.conv2d(self.eval_X, self.conv1_w, strides=[1, 1, 1, 1], padding='SAME')

        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, self.conv1_b))
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 2D Convolution Layer, then Bias + ReLU, then Pooling Layer
        conv2 = tf.nn.conv2d(pool1, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.conv2_b))
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Reshape 4D Pool Tensor into a 2D Tensor
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

        # Fully Connected Layer 1 -> ReLU Activation
        hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_w) + self.fc1_b)

        # Add dropout --> Only during training
        if train:
            hidden = tf.nn.dropout(hidden, 0.5)

        # Fully Connected Layer 2 -> for softmax (actual softmax performed in loss function)
        return tf.matmul(hidden, self.fc2_w) + self.fc2_b

    def loss(self):
        """
        Build the computation graph for calculating the cross entropy loss, and add on the L2
        Regularization of the Feed-Forward layers.
        """
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.Y))
        l2_regularizer = (tf.nn.l2_loss(self.fc1_w) + tf.nn.l2_loss(self.fc1_b) + tf.nn.l2_loss(
            self.fc2_w) + tf.nn.l2_loss(self.fc2_b))

        # Add L2 Loss * Discount Factor to Cross-Entropy Loss
        loss += self.l2_discount * l2_regularizer

        # Add a summary to track Loss over time
        tf.scalar_summary('Loss', loss)

        return loss

    def train(self):
        """
        Build the computation graph for computing gradients and doing backprop.
        """
        # Set up a variable that's incremented once per batch and controls the learning_rate decay.
        batch = tf.Variable(0)

        # Decay once per epoch, using exponential schedule
        learning_rate = tf.train.exponential_decay(self.lr, batch * self.batch_size,
                                                   self.train_size, self.lr_decay, staircase=True)
        # Add a summary to track learning rate over time
        tf.scalar_summary('Learning_Rate', learning_rate)

        # Use simple momentum optimizer
        train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.loss_val,
                                                                           global_step=batch)
        return learning_rate, train_op


def init_weight(shape, name):
    """
    Initialize a Tensor corresponding to a weight matrix with the given shape and name.

    :param shape: Shape of the weight tensor.
    :param name: Name of the weight tensor in the computation graph.
    :return: Tensor object with given shape and name, initialized from a standard normal.
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


def init_bias(shape, value, name):
    """
    Initialize a Tensor corresponding to a bias vector with the given shape and name.

    :param shape: Shape of the bias vector (as an int, not a list).
    :param value: Value to initialize bias to.
    :param name: Name of the bias vector in the computation graph.
    :return: Tensor (Vector) object with given shape and name, initialized with given bias.
    """
    return tf.Variable(tf.constant(value, shape=[shape]), name=name)