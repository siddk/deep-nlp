"""
train.py

Core runner for training the LSTM-Based Language Model. Feeding in one sentence at a time,
the network tries to predict the next word using a series of stacked LSTM layers.
"""
from preprocessor.reader import *
import tensorflow as tf
import time
import os

# Set up Tensorflow Training Parameters, or "FLAGS"
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir', 'log/checkpoints', "Directory where to write checkpoints.")
tf.app.flags.DEFINE_string('summary_dir', 'log/summaries', 'Directory where to write summaries.')
tf.app.flags.DEFINE_integer('epochs', 10, "Maximum number of epochs to run.")
tf.app.flags.DEFINE_integer('batch_size', 20, "Number of examples to process in a batch.")
tf.app.flags.DEFINE_integer('window_size', 20, "Size of the window fed to the LSTM Network.")
tf.app.flags.DEFINE_integer('num_layers', 3, "Number of stacked LSTM Layers.")
tf.app.flags.DEFINE_float('learning_rate', 0.5, "Learning rate for SGD.")
tf.app.flags.DEFINE_integer('embedding_size', 20, "Dimension of the embeddings.")
tf.app.flags.DEFINE_integer('hidden_size', 60, "Dimension of the hidden layers.")
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

    x, y = build_iterators(x, y)
    test_x, test_y = build_iterators(x, y, 1, 1)




if __name__ == "__main__":
    tf.app.run()