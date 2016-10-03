"""
reader.py

Read in a source file, build a vocabulary, vectorize the data, return X, Y tuples.
"""
import numpy as np
import cPickle as pickle
import re
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

START = ">>>START<<<"
STOP = ">>>STOP<<<"
UNK = ">>>UNK<<<"
UNK_ID = 2
START_VOCAB = [START, STOP, UNK]
WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")


def basic_tokenizer(sentence):
    """
    Very basic tokenizer: split the sentence into a list of tokens, lowercase.
    """
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(WORD_SPLIT, space_separated_fragment))
    return [w.lower() for w in words if w]


def read(lang_id, src, vocab_size, window_size, num_train, num_val, num_test):
    """
    Read in a source file, build vocabulary, vectorize data within fixed window.

    :param lang_id: Language identifier.
    :param src: Source file path.
    :param vocab_size: Size of the vocabulary.
    :param window_size: Size of the language model window.
    :param num_train: Number of training examples.
    :param num_val: Number of validation examples.
    :param num_test: Number of test examples.
    :return: Tuple (X, Y, valX, valY, testX, testY) consisting of training and test data.
    """
    all_words = build_vocabulary(lang_id, src)
    vocab, _ = init_vocab(lang_id, vocab_size)

    train_x, train_y = [], []
    rolling_window = window_size + 1
    for ptr in range(num_train):
        start, end = ptr * rolling_window, (ptr * rolling_window) + window_size
        train_x.append(map(lambda x: vocab.get(x, UNK_ID), all_words[start:end]))
        train_y.append(vocab.get(all_words[end], UNK_ID))
    train_x, train_y = np.array(train_x), np.array(train_y)
    print "Training Shapes: X:", train_x.shape, "Y:", train_y.shape
    assert(len(train_x) == len(train_y))
    with open(os.path.join(FLAGS.train_dir, lang_id + ".pik"), 'wb') as f:
        pickle.dump((train_x, train_y), f)

    val_x, val_y = [], []
    for ptr in range(num_train, num_train + num_val):
        start, end = ptr * rolling_window, (ptr * rolling_window) + window_size
        val_x.append(map(lambda x: vocab.get(x, UNK_ID), all_words[start:end]))
        val_y.append(vocab.get(all_words[end], UNK_ID))
    val_x, val_y = np.array(val_x), np.array(val_y)
    print "Validation Shapes: X:", val_x.shape, "Y:", val_y.shape
    assert(len(val_x) == len(val_y))
    with open(os.path.join(FLAGS.val_dir, lang_id + ".pik"), 'wb') as f:
        pickle.dump((val_x, val_y), f)

    test_x, test_y = [], []
    for ptr in range(num_train + num_val, num_train + num_val + num_test):
        start, end = ptr * rolling_window, (ptr * rolling_window) + window_size
        test_x.append(map(lambda x: vocab.get(x, UNK_ID), all_words[start:end]))
        test_y.append(vocab.get(all_words[end], UNK_ID))
    test_x, test_y = np.array(test_x), np.array(test_y)
    print "Test Shapes: X:", test_x.shape, "Y:", test_y.shape
    assert(len(test_x) == len(test_y))
    with open(os.path.join(FLAGS.test_dir, lang_id + ".pik"), 'wb') as f:
        pickle.dump((test_x, test_y), f)

    return train_x, train_y, val_x, val_y, test_x, test_y


def build_vocabulary(lang_id, src):
    """
    Build the vocabulary using the most frequent words for the language.

    :param lang_id: Language identifier.
    :param src: Source file.
    :return: List of all words, with START/STOP tokens, as a single long list.
    """
    vocab, all_words = {}, []
    with tf.gfile.GFile(src, 'rb') as f:
        for line in f:
            words = basic_tokenizer(line)
            for w in words:
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
            all_words += [START] + words + [STOP]

    vocab_list = START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    vocab_path = lang_id + ".vocab"
    with tf.gfile.GFile(os.path.join(FLAGS.vocab_dir, vocab_path), 'wb') as f:
        for w in vocab_list:
            f.write(w + b"\n")

    return all_words


def init_vocab(language_id, vocab_size):
    """
    Given the language_id, load and return vocabulary dictionary, up to vocab_size.

    :param language_id: Language identifier
    :param vocab_size: Max vocabulary size.
    :return: Tuple of vocab2idx, idx2vocab
    """
    rev_vocab = []
    with tf.gfile.GFile(os.path.join(FLAGS.vocab_dir, language_id + ".vocab"), mode="rb") as f:
        rev_vocab.extend(f.readlines()[:vocab_size])
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab


def load_train_data(lang_id):
    """
    Given the language id, load the respective train data.

    :param lang_id: Language identifier
    :return: Tuple of x, y data
    """
    with open(os.path.join(FLAGS.train_dir, lang_id + ".pik"), 'rb') as f:
        return pickle.load(f)


def load_test_data(lang_id):
    """
    Given the language id, load the respective test data.

    :param lang_id: Language identifier
    :return: Tuple of x, y data
    """
    with open(os.path.join(FLAGS.test_dir, lang_id + ".pik"), 'rb') as f:
        return pickle.load(f)