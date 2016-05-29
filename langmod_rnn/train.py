"""
train.py

Core runner for training the LSTM-Based Language Model. Feeding in one sentence at a time,
the network tries to predict the next word using a series of stacked LSTM layers.
"""
from model.langmod_lstm import LangmodLSTM
from preprocessor.reader import *
import tensorflow as tf
import time
import os

# Set up Tensorflow Training Parameters, or "FLAGS"
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir', 'log/checkpoints', "Directory where to write checkpoints.")
tf.app.flags.DEFINE_string('summary_dir', 'log/summaries', 'Directory where to write summaries.')
tf.app.flags.DEFINE_integer('decay_epochs', 10, "Maximum number of epochs to run before decay.")
tf.app.flags.DEFINE_integer('epochs', 40, 'Maximum maximum number of epochs to run.')
tf.app.flags.DEFINE_integer('batch_size', 20, "Number of examples to process in a batch.")
tf.app.flags.DEFINE_integer('window_size', 20, "Size of the window fed to the LSTM Network.")
tf.app.flags.DEFINE_integer('num_layers', 3, "Number of stacked LSTM Layers.")
tf.app.flags.DEFINE_float('learning_rate', 1.0, "Learning rate for SGD.")
tf.app.flags.DEFINE_float('lr_decay', 0.8, "Learning rate decay (discount after each epoch.)")
tf.app.flags.DEFINE_float('max_grad_norm', 10, "Maximum gradient norm for clipping gradients.")
tf.app.flags.DEFINE_integer('embedding_size', 100, "Dimension of the embeddings.")
tf.app.flags.DEFINE_string('train_path', 'data/english-senate-0.txt', "Path to train data.")
tf.app.flags.DEFINE_string('test_path', 'data/english-senate-2.txt', "Path to test data.")
tf.app.flags.DEFINE_integer('vocab_size', 5000, "Size of the vocabulary.")


def main(_):
    """
    Main Training function, loads and preprocesses data, builds a batch iterator, instantiates
    model, starts training process.
    """
    # Fix directories
    if tf.gfile.Exists(FLAGS.summary_dir):
        tf.gfile.DeleteRecursively(FLAGS.summary_dir)
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.summary_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Load data (as iterators)
    x, y, vocab = encode_raw_data(FLAGS.train_path)
    test_x, test_y, _ = encode_raw_data(FLAGS.test_path, vocab)

    # Start training
    with tf.Session() as sess:
        # Instantiate Network
        langmod = LangmodLSTM(FLAGS.batch_size, FLAGS.window_size)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build summary operations
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.summary_dir + "/train", sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summary_dir + "/test")

        # Initialize all variables
        sess.run(tf.initialize_all_variables())
        bsz = FLAGS.batch_size

        for epoch in range(FLAGS.epochs):
            print "On Epoch", epoch + 1, "of", FLAGS.epochs
            x_iter, y_iter = build_iterators(x, y)
            test_x_iter, test_y_iter = build_iterators(test_x, test_y, 1, 1)

            lr_decay = FLAGS.lr_decay ** max(epoch - FLAGS.decay_epochs, 0.0)
            langmod.assign_lr(sess, FLAGS.learning_rate * lr_decay)


if __name__ == "__main__":
    tf.app.run()