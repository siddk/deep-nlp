"""
train.py

Core runner for training the Deep CIFAR-10 model. Builds a small-ish Convolutional
Neural Network for classifying images into one of 10 different categories: airplanes, automobiles,
birds, cats, deer, dogs, frogs, horses, ships, and trucks.

Note: Requires several hours to train on a GPU. Can also be run on a CPU (tensorflow takes care of
the underlying changes).

Credit: Tensorflow CIFAR Tutorial: http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

from model.cifar import CIFAR

# Set up Tensorflow Training Parameters, or "FLAGS"
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'log/cifar_train',
                           "Directory where to write event logs and checkpoints.")
tf.app.flags.DEFINE_integer('max_steps', 1000000, "Maximum number of batches to run.")
tf.app.flags.DEFINE_boolean('log_device_placement', False, "Whether to log device placement.")
tf.app.flags.DEFINE_integer('batch_size', 128, "Number of images to process in a batch.")
tf.app.flags.DEFINE_string('data_dir', 'log/cifar10_data', "Path to the CIFAR-10 data directory.")


def train():
    """
    Core Training Function, builds the CIFAR Model, and runs the training logic.
    """
    # Start a Tensorflow Computation Graph
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Read in the Training Data, apply distortions
        images, labels = CIFAR.distorted_inputs()

        # Instantiate a CIFAR Model with inference, loss, and train_ops
        cifar = CIFAR(images, labels, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        # Build a Summary Writer
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([cifar.train_op, cifar.loss_value])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    """
    Main Function, called by tf.app.run(). Downloads the Training Data if Necessary, performs
    all preprocessing, then builds and trains the model.
    """
    # Download the Training Data
    CIFAR.download_and_extract()

    # Reset Logging Directory
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    # Train
    train()

# Python Main function, starts the tensorflow session by invoking main() [see above]
if __name__ == "__main__":
    tf.app.run()