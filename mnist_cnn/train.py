"""
train.py

Core file for training a LeNet-5 Convolutional Neural Network for classifying handwritten digits in
the MNIST data set. Feeding in an image of a handwritten number, the network tries to predict which
digit is depicted.

Credit: Tensorflow MNIST CNN Tutorial
"""
from __future__ import division
from model.mnist_cnn import MnistCNN
from preprocessor.loader import *
from six.moves import xrange
import numpy as np
import sys
import tensorflow as tf
import time

# Set up Tensorflow Training Parameters, or "FLAGS"
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('source_url', 'http://yann.lecun.com/exdb/mnist/', 'Data Location.')
tf.app.flags.DEFINE_string('log_dir', 'log/checkpoints', "Directory where to write checkpoints.")
tf.app.flags.DEFINE_string('summary_dir', 'log/summaries', 'Directory where to write summaries.')
tf.app.flags.DEFINE_string('data_dir', 'data/', 'Directory where to store data.')
tf.app.flags.DEFINE_integer('image_size', 28, 'Size of the image (width/length).')
tf.app.flags.DEFINE_integer('num_channels', 1, 'Number of image channels.')
tf.app.flags.DEFINE_float('pixel_depth', 255., 'Depth of pixel color.')
tf.app.flags.DEFINE_integer('hidden_size', 512, 'Size of the hidden feed-forward layer.')
tf.app.flags.DEFINE_float('l2_discount', .0005, 'Factor to discount L2 Regularization by.')
tf.app.flags.DEFINE_float('learning_rate', .01, 'Base learning rate to start with.')
tf.app.flags.DEFINE_float('lr_decay', 0.95, 'Learning rate decay rate.')
tf.app.flags.DEFINE_integer('epochs', 10, 'Number of epochs for training.')
tf.app.flags.DEFINE_integer('num_labels', 10, 'Number of digit classes.')
tf.app.flags.DEFINE_integer('train_size', 55000, 'Size of the training set.')
tf.app.flags.DEFINE_integer('test_size', 10000, 'Size of the test set.')
tf.app.flags.DEFINE_integer('validation_size', 5000, 'Size of the validation set.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Number of images to process per train step.')
tf.app.flags.DEFINE_integer('eval_frequency', 100, 'Number of steps (batches) between evaluations.')


def eval_in_batches(data, model, sess):
    """
    Get all predictions for a large dataset by running it in small batches.

    :param data: Data to evaluate
    :param model: MNIST_CNN Model instance
    :param sess: Tensorflow session to evaluate Variables in.
    """
    size = data.shape[0]
    if size < FLAGS.batch_size:
        raise ValueError("Batch size for evaluation larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, FLAGS.num_labels), dtype=numpy.float32)
    for begin in xrange(0, size, FLAGS.batch_size):
        end = begin + FLAGS.batch_size
        if end <= size:
            predictions[begin:end, :] = sess.run(model.eval_prediction,
                                                 feed_dict={model.eval_X: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(model.eval_prediction,
                                         feed_dict={model.eval_X: data[-FLAGS.batch_size:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


def main(_):
    """
    Main training function, loads and extracts MNIST Data from Yann LeCun's Website, builds the
    model, and performs training and evaluation.
    """
    # Fix directories
    if tf.gfile.Exists(FLAGS.summary_dir):
        tf.gfile.DeleteRecursively(FLAGS.summary_dir)
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.summary_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Load Data
    x, y, val_x, val_y, test_x, test_y = load_data()

    # Start Session
    start_time = time.time()
    with tf.Session() as sess:
        # Instantiate network
        mnist_cnn = MnistCNN()

        # Create a Saver
        saver = tf.train.Saver(tf.all_variables())

        # Build summary operations
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.summary_dir + "/train", sess.graph)

        # Initialize all variables
        sess.run(tf.initialize_all_variables())
        print "Variables Initialized!"
        print ""
        bsz = FLAGS.batch_size

        # Loop through training steps.
        for step in xrange(int(FLAGS.epochs * FLAGS.train_size) // bsz):
            # Compute the offset of the current minibatch in the data.
            offset = (step * bsz) % (FLAGS.train_size - bsz)
            batch_x = x[offset:(offset + bsz), ...]
            batch_y = y[offset:(offset + bsz)]

            # Run the graph and fetch some of the nodes.
            _, l, lr, train_acc, tr_summary = sess.run(
                [mnist_cnn.train_op, mnist_cnn.loss_val, mnist_cnn.learning_rate,
                 mnist_cnn.train_acc, merged], feed_dict={mnist_cnn.X: batch_x,
                                                          mnist_cnn.Y: batch_y})

            if step % FLAGS.eval_frequency == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                val_preds = eval_in_batches(val_x, mnist_cnn, sess)
                val_acc = np.sum(np.argmax(val_preds, 1) == val_y) / val_y.shape[0]

                print 'Step %d (epoch %.2f), %.1f ms' % (step, float(step) * bsz / FLAGS.train_size,
                                                         1000 * elapsed_time / FLAGS.eval_frequency)
                print 'Minibatch loss: %.3f, learning rate: %.6f' % (l, lr)
                print 'Minibatch accuracy: %.3f%%' % train_acc
                print 'Validation accuracy: %.3f%%' % val_acc
                print ""

                # Add summaries
                train_writer.add_summary(tr_summary, step)

                sys.stdout.flush()

            if (step * bsz / FLAGS.train_size) == 0:
                # Save the model
                checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, step)

        # Finally print the result!
        test_predictions = eval_in_batches(test_x, mnist_cnn, sess)
        test_accuracy = np.sum(np.argmax(test_predictions, 1) == test_y) / test_y.shape[0]
        print 'Test accuracy: %.3f%%' % test_accuracy

if __name__ == "__main__":
    tf.app.run()