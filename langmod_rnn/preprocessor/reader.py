"""
reader.py

Core preprocessing file, loads raw data, converts into numeric labels (not one-hot, hot-encoded
labels), builds iterator over each batch depending on the Batch Size and the Window Size.
"""
import collections
import numpy as np
import tensorflow as tf

EOS = "<eos>"
UNK = "<unk>"
FLAGS = tf.app.flags.FLAGS


def encode_raw_data(data_path, vocab=None):
    """
    Read in a raw data file, build vocab, add EOS and UNK Tokens, and return the label-hot
    encoded data.

    :param data_path: Path to raw data file (train/test).
    :param vocab: Vocabulary (if built).
    :return: Tuple consisting of label-Hot Encoded data and vocabulary
    """
    # Read data, add STOP symbols between sentences
    with tf.gfile.GFile(data_path, 'r') as f:
        data = [EOS] + f.read().replace("\n", EOS + " ").split()

    if not vocab:
        # Build Word Counts for UNK'ing least-common words
        word_counts = collections.Counter(data)
        word_counts = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))

        # Build Vocab
        vocab = collections.defaultdict(int)
        for i in range(1, FLAGS.vocab_size):
            vocab[word_counts[i][0]] = i

    # Hot encode the entire set of raw data
    x, y = map(lambda w: vocab[w], data[:-1]), map(lambda w: vocab[w], data[1:])
    return x, y, vocab


def build_iterators(x, y, batch_size=None, window_size=None):
    """
    Build batch iterators given raw data and labels.

    :param x: Raw data (hot-encoded)
    :param y: Raw labels (time-shifted data hot-encoded)
    :param batch_size: Number of examples per batch.
    :param window_size: Number of words per example.
    :return: Tuple of data, label iterators
    """
    # Update batch_size based on function parameters
    if not batch_size:
        batch_size, window_size = FLAGS.batch_size, FLAGS.window_size

    data_len = len(x) / (batch_size * window_size)
    x, y = x[:data_len], y[:data_len]
    batch_len = len(x) / batch_size

    # Allocate data, label tensors
    data, labels = np.zeros([batch_size, batch_len]), np.zeros([batch_size, batch_len])

    for i in range(batch_size):
        start, end = batch_len * i, batch_len * (i + 1)
        data[i], labels[i] = x[start:end], y[start:end]

    epoch_size = batch_len / window_size
    for i in range(epoch_size):
        start, end = i * window_size, (i + 1) * window_size
        x, y = data[:, start:end], labels[:, start:end]
        yield x, y

