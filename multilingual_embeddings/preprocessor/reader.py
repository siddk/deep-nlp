"""
reader.py

Read and load data, build vocabularies, write output token files.
"""
import os
import re
import tensorflow as tf
from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS

# Regular expressions used to tokenize.
WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
DIGIT_RE = re.compile(br"\d")

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


def basic_tokenizer(sentence):
    """
    Very basic tokenizer: split the sentence into a list of tokens, lowercase.
    """
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(WORD_SPLIT, space_separated_fragment))
    return [w.lower() for w in words if w]


def build_buckets(buckets, max_vocab, source_languages, target_languages):
    """
    Build bucketed versions of the data, tokenizing it and writing it to a file in the process.

    :param buckets: List of pairs representing (source, target) bucket length.
    :param max_vocab: Dictionary mapping language id to Maximum Vocabulary Size.
    """
    if len(os.listdir(os.path.join(FLAGS.bucket_dir, str(0)))) <= 1:
        for target_id in target_languages:
            target_vocab, _ = init_vocab(target_id, max_vocab[target_id])
            for source_id in source_languages:
                source_vocab, _ = init_vocab(source_id, max_vocab[source_id])
                base_fp = "%s-%s." % tuple(sorted([source_id, target_id]))
                eval_fp = "%s-%s-eval." % tuple(sorted([source_id, target_id]))
                print base_fp, source_id, target_id
                data_set = {k: [] for k in range(len(buckets))}
                with gfile.GFile(os.path.join(FLAGS.raw_dir, base_fp + source_id), 'r') as src:
                    with gfile.GFile(os.path.join(FLAGS.raw_dir, base_fp + target_id), 'r') as trg:
                        source, target = src.readline(), trg.readline()
                        while source and target:
                            src_ids = [source_vocab.get(w, UNK_ID) for w in basic_tokenizer(source)]
                            trg_ids = [target_vocab.get(w, UNK_ID) for w in basic_tokenizer(target)]
                            trg_ids.append(EOS_ID)

                            # Find a bucket
                            for bucket_id, (source_size, target_size) in enumerate(buckets):
                                if len(src_ids) < source_size and len(trg_ids) < target_size:
                                    data_set[bucket_id].append([src_ids, trg_ids])
                                    break

                            source, target = src.readline(), trg.readline()
                for k in data_set:
                    counter = 0
                    data_train = data_set[k][:-200]
                    data_eval = data_set[k][-200:]
                    bucket_path = os.path.join(FLAGS.bucket_dir, str(k))
                    with gfile.GFile(os.path.join(bucket_path, base_fp + source_id), 'w') as src:
                        with gfile.GFile(os.path.join(bucket_path, base_fp + target_id), 'w') as trg:
                            for i in range(len(data_train)):
                                counter += 1
                                s, t = data_train[i]
                                src.write(" ".join(map(str, s)) + "\n")
                                trg.write(" ".join(map(str, t)) + "\n")
                    with gfile.GFile(os.path.join(bucket_path, eval_fp + source_id), 'w') as src:
                        with gfile.GFile(os.path.join(bucket_path, eval_fp + target_id), 'w') as trg:
                            for i in range(len(data_eval)):
                                s, t = data_eval[i]
                                src.write(" ".join(map(str, s)) + "\n")
                                trg.write(" ".join(map(str, t)) + "\n")
                    print "Bucket", k, "for %s-%s" % (source_id, target_id), "has", counter, "examples!"


def build_vocabularies(source_languages, target_languages):
    """
    Build vocabulary dictionaries for each language pair in source -> target.

    :param source_languages: List of source language identifiers
    :param target_languages: List of target language identifiers
    """
    if len(os.listdir(FLAGS.vocab_dir)) <= 1:
        language_paths = {k: [] for k in source_languages + target_languages}
        for source in source_languages:
            for target in target_languages:
                path = "%s-%s." % tuple(sorted([source, target]))
                language_paths[source].append(path + source)
                language_paths[target].append(path + target)

        for k in language_paths:
            build_vocab(language_paths[k], k)


def build_vocab(language_paths, language_id):
    """
    Build a vocabulary for the language using

    :param language_paths: Files to read from, to build vocabulary.
    :return: Tuple of vocab2idx, idx2vocab
    """
    vocab = {}
    for file_name in language_paths:
        with tf.gfile.GFile(os.path.join(FLAGS.raw_dir, file_name), 'rb') as f:
            for line in f:
                words = basic_tokenizer(line)
                for w in words:
                    if w in vocab:
                        vocab[w] += 1
                    else:
                        vocab[w] = 1
    vocab_list = START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    vocab_path = language_id + ".vocab"
    with gfile.GFile(os.path.join(FLAGS.vocab_dir, vocab_path), 'wb') as f:
        for w in vocab_list:
            f.write(w + b"\n")
    print "Length of Vocabulary for", language_id, "is:", len(vocab_list)


def init_vocab(language_id, vocab_size):
    """
    Given the language_id, load and return vocabulary dictionary, up to vocab_size.

    :param language_id: Language identifier
    :param vocab_size: Max vocabulary size.
    :return: Tuple of vocab2idx, idx2vocab
    """
    rev_vocab = []
    with gfile.GFile(os.path.join(FLAGS.vocab_dir, language_id + ".vocab"), mode="rb") as f:
        rev_vocab.extend(f.readlines()[:vocab_size])
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab