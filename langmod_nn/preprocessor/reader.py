"""
reader.py

Core preprocessor for the Deep Bigram Language Model. Reads in data from file, and converts
to integer index in vocabulary.
"""
import collections
import numpy as np
import tensorflow as tf

EOS = "<eos>"
UNK = "<unk>"
FLAGS = tf.app.flags.FLAGS


def get_bigrams(data_path, vocab=None):
    """
    Vectorize a data file, turning it into bigram pairs of integers.

    :param data_path: Path to data.
    :return: X, Y pair consisting of first word, second word encoded label-hot.
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

    # Build Bigrams
    x, y = map(lambda w: vocab[w], data[:-1]), map(lambda w: vocab[w], data[1:])
    x = x[:((len(x) / FLAGS.batch_size) * FLAGS.batch_size)]
    y = y[:len(x)]

    return np.array(x, dtype='int64'), np.array(y, dtype='int64'), vocab







