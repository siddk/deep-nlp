"""
cifar.py

Core class defining the CIFAR Convolutional Neural Network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
from preprocessor import cifar10_input
import tensorflow as tf

# Get current parameters
FLAGS = tf.app.flags.FLAGS

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# Data URL
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


class CIFAR:
    def __init__(self, images, labels, global_step, train=True):
        """
        Instantiate a CIFAR Model, and build the computation pipelines for

        :param images: Tensor of preprocessed images.
        :param labels: Labels for aforementioned images.
        :param global_step: Step size.
        :param train: Boolean if training or evaluation model.
        """
        self.images, self.labels = images, labels
        self.global_step = global_step

        # Training pipeline
        if train:
            # Build a Graph that computes the logits predictions (running an image through model)
            self.logits = self.inference()

            # Build a Graph that computes the loss value from the logits, and the true labels
            self.loss_value = self.loss()

            # Build the Backpropagation Graph (i.e. compute Gradients from loss, etc.)
            self.train_op = self.train()

        # Evaluation pipeline
        if not train:
            # Build a Graph that computes the logits predictions (running an image through model)
            self.logits = self.inference()

            # Calculate predictions.
            self.top_k_op = tf.nn.in_top_k(self.logits, self.labels, 1)


    def inference(self):
        """
        Build the CIFAR-10 Model.

        :return: Tensor of unnormalized logits.
        """
        # Conv1 Layer
        with tf.variable_scope('conv1') as scope:
            kernel = CIFAR._variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                                       stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(self.images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = CIFAR._variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            CIFAR._activation_summary(conv1)

        # Pool1 Layer
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',
                               name='pool1')
        # Norm1 Layer
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        # Conv2 Layer
        with tf.variable_scope('conv2') as scope:
            kernel = CIFAR._variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                                       stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = CIFAR._variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)
            CIFAR._activation_summary(conv2)

        # Norm2 Layer
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        # Pool2 Layer
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',
                               name='pool2')

        # Local3 Layer (Dense Feed-Forward Layer)
        with tf.variable_scope('local3') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = CIFAR._variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04,
                                                        wd=0.004)
            biases = CIFAR._variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            CIFAR._activation_summary(local3)

        # Local4 Layer (Dense Feed-Forward Layer)
        with tf.variable_scope('local4') as scope:
            weights = CIFAR._variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04,
                                                        wd=0.004)
            biases = CIFAR._variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
            CIFAR._activation_summary(local4)

        # Softmax Layer, i.e. softmax(WX + b) --> Doesn't actually perform softmax, handled in loss
        with tf.variable_scope('softmax_linear') as scope:
            weights = CIFAR._variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                                        stddev=1 / 192.0, wd=0.0)
            biases = CIFAR._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
            CIFAR._activation_summary(softmax_linear)

        return softmax_linear

    def loss(self):
        """
        Compute the Loss Tensor, which is comprised of the logits cross-entropy loss with the
        true labels, plus all of the L2 Loss terms from each of the weights (for regularization).
        Also add summary for "Loss" and "Loss/avg"

        :return: Loss tensor of type float.
        """
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(self.labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, labels,
                                                                name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def train(self):
        """
        Train CIFAR-10 model.

        Create an optimizer and apply to all trainable variables. Add moving average for all
        trainable variables.

        :return train_op: op for training.
        """
        # Variables that affect learning rate.
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, self.global_step, decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR, staircase=True)
        tf.scalar_summary('learning_rate', lr)

        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = CIFAR._add_loss_summaries(self.loss_value)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(lr)
            grads = opt.compute_gradients(self.loss_value)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, self.global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op

    @staticmethod
    def _activation_summary(x):
        """
        Helper to create summaries for activations.

        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.

        :param x: Tensor
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on Tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.histogram_summary(tensor_name + '/activations', x)
        tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    @staticmethod
    def _add_loss_summaries(total_loss):
        """
        Add summaries for losses in CIFAR-10 model.
        Generates moving average for all losses and associated summaries for visualizing the
        performance of the network.

        :param total_loss: Total loss from loss().
        :return loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.scalar_summary(l.op.name + ' (raw)', l)
            tf.scalar_summary(l.op.name, loss_averages.average(l))

        return loss_averages_op

    @staticmethod
    def _variable_on_cpu(name, shape, initializer):
        """
        Helper to create a Variable stored on CPU memory.

        :param name: Name of the variable
        :param shape: List of integers corresponding to shape
        :param initializer: Initializer for Variable
        :return Variable Tensor
        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    @staticmethod
    def _variable_with_weight_decay(name, shape, stddev, wd):
        """
        Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.

        :param name: Name of the variable
        :param shape: List of integers corresponding to shape
        :param stddev: Standard deviation of a truncated Gaussian
        :param wd: Add L2Loss weight decay multiplied by this float. If None, weight decay is not
                   added for this Variable
        :return Variable Tensor
        """
        var = CIFAR._variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    @staticmethod
    def inputs(eval_data):
        """
        Construct regular input for CIFAR evaluation.

        :param eval_data: bool, indicating if one should use the train or eval data set.
        :return: Tuple of images, labels where:
            images: 4D tensor of shape (batch_size, IMAGE_SIZE, IMAGE_SIZE, 3) -> Normal RGB Image
            labels: 1D tensor of shape (batch_size) -> Each label is the number of class from 0 - 9
        """
        if not FLAGS.data_dir:
            raise ValueError('Please supply a data_dir')
        data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
        return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                                    batch_size=FLAGS.batch_size)

    @staticmethod
    def distorted_inputs():
        """
        Construct distorted input for CIFAR training.

        :return: Tuple of images, labels where:
            images: 4D tensor of shape (batch_size, IMAGE_SIZE, IMAGE_SIZE, 3) -> Normal RGB Image
            labels: 1D tensor of shape (batch_size) -> Each label is the number of class from 0 - 9
        """
        if not FLAGS.data_dir:
            raise ValueError('Please supply a data_dir')
        data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
        return cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)

    @staticmethod
    def download_and_extract():
        """
        Download and Extract the Training Data, writing it to FLAGS.data_dir
        """
        dest_directory = FLAGS.data_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)

        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)

        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                 float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)
