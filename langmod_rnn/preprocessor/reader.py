"""
reader.py

Core utility file for reading, parsing, and vectorizing the training/test data.
"""
import collections
import numpy as np
import tensorflow as tf

# Get Tensorflow parameters
FLAGS = tf.app.flags.FLAGS
UNK = "<unk>"
EOS = "<eos>"


def get_examples(data_path, vocab=None):
    """
    Load data from data_path, parse and vectorize depending on the vocab, then return tuple of
    data and labels.

    :param data_path: Path to data.
    :param vocab: Vocabulary to use (UNK if not in vocab).
    :return: Tuple of (X, Y) where the shape of X/Y is [num_batches, batch_size, window_size]
    """
    with tf.gfile.GFile(data_path, 'r') as f:
        words = f.read().replace("\n", EOS + " ").split()

    if not vocab:
        counter = collections.Counter(words)
        count_pairs = sorted(counter.items())
        w, _ = list(zip(*count_pairs))
        vocab = collections.defaultdict(int)
        for i in range(1, FLAGS.vocab_size):
            vocab[w[i]] = i

    chunk_size = FLAGS.batch_size * FLAGS.window_size
    words = words[:len(words) - (len(words) % chunk_size) + 1]
    X, Y = [], []
    for i in range(len(words) / chunk_size):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk_x, chunk_y = words[start:end], words[start + 1:end + 1]
        X.append([vocab[wi] for wi in chunk_x])
        Y.append([vocab[wi] for wi in chunk_y])

    X, Y = np.array(X, dtype=np.int32), np.array(Y, dtype=np.int32)
    X = np.reshape(X, [len(X), FLAGS.batch_size, FLAGS.window_size])
    Y = np.reshape(Y, [len(Y), FLAGS.batch_size, FLAGS.window_size])

    return X, Y, vocab


