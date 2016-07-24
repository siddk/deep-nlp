"""
reader.py

Core utility script, loads and parses the French and English train and test data.
"""
import os
import tensorflow as tf
from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS

# Special vocabulary symbols - we always put them at the start.
PAD = b"<PAD>"
GO = b"<GO>"
EOS = b"<EOS>"
UNK = b"<UNK>"
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
START_VOCAB = [PAD, GO, EOS, UNK]


def load_data():
    """
    Reads the raw data, builds and writes the vocabulary, then builds and writes the tokens for
    both the train and test sets.
    """
    # Setup Path Variables
    fr_train_path = os.path.join(FLAGS.data_dir, "raw/europarl-v7.fr-en.fr")
    en_train_path = os.path.join(FLAGS.data_dir, "raw/europarl-v7.fr-en.en")
    fr_test_path = os.path.join(FLAGS.data_dir, "raw/french-senate-2.txt")

    # Build vocabularies
    fr_vocab_path = os.path.join(FLAGS.data_dir, "vocabulary/fr.vocab")
    en_vocab_path = os.path.join(FLAGS.data_dir, "vocabulary/en.vocab")

    create_vocabulary(fr_train_path, fr_vocab_path)
    create_vocabulary(en_train_path, en_vocab_path)

    # Build tokens
    fr_train_tokens = os.path.join(FLAGS.data_dir, "tokens/fr.train")
    en_train_tokens = os.path.join(FLAGS.data_dir, "tokens/en.train")
    fr_test_tokens = os.path.join(FLAGS.data_dir, "tokens/fr.test")

    get_tokens(fr_train_path, fr_train_tokens, fr_vocab_path)
    get_tokens(en_train_path, en_train_tokens, en_vocab_path)
    get_tokens(fr_test_path, fr_test_tokens, fr_vocab_path)

    return fr_vocab_path, en_vocab_path, fr_train_tokens, en_train_tokens, fr_test_tokens


def get_tokens(data_path, token_path, vocab_path):
    """
    Read a file and write a token id file using the provided vocabulary.

    :param data_path: Path to data to tokenize.
    :param token_path: Path where to write tokens.
    :param vocab_path: Path to vocabulary.
    """
    if not gfile.Exists(token_path):
        vocab, _ = init_vocab(vocab_path)
        with gfile.GFile(data_path, 'rb') as data_file:
            with gfile.GFile(token_path, 'wb') as token_file:
                for line in data_file:
                    token_ids = [vocab.get(w, UNK_ID) for w in line.split()]
                    token_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def create_vocabulary(data_path, vocab_path):
    """
    Given path to data, and a place to write the vocabulary, build the vocabulary using the
    most frequent max_vsz words.

    :param data_path: Path to data to vocabularize.
    :param vocab_path: Path to location to write vocabulary.
    """
    if not gfile.Exists(vocab_path):
        vocab = {}
        with gfile.GFile(data_path, mode='rb') as f:
            for line in f:
                tokens = line.split()
                for w in tokens:
                    if w in vocab:
                        vocab[w] += 1
                    else:
                        vocab[w] = 1
        vocab_list = START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        vocab_list = vocab_list[:FLAGS.max_vsz]
        with gfile.GFile(vocab_path, 'wb') as f:
            for w in vocab_list:
                f.write(w + b"\n")


def init_vocab(vocab_path):
    """
    Give path to written vocabulary, load and return dictionaries containing both the string -> id
    and the id -> string mappings.

    :param vocab_path: Path to written vocabulary.
    :return: Tuple of string -> id vocab, id -> string vocab
    """
    if gfile.Exists(vocab_path):
        rev_vocab = []
        with gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError('Vocabulary Path does not Exist!')