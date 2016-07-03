"""
reader.py

Core util file for preprocessing bAbI task data.
"""
import numpy as np
import os
import re


def load_task(data_dir, task_id):
    """
    Load the given task from the data directory, then parse and return
    the train and test data as a tuple.

    :param data_dir: Path to task data.
    :param task_id: Task to load data for.
    :return: Tuple of train, test data.
    """
    train_file = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                  if 'qa{}_'.format(task_id) in f and 'train' in f]
    test_file = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                 if 'qa{}_'.format(task_id) in f and 'test' in f]
    train, test = get_stories(train_file[0]), get_stories(test_file[0])
    return train, test


def get_stories(filepath):
    """
    Given a path to task data, load, parse, and return the Stories, Questions,
    and Answers.

    :param filepath: Path to task data.
    :return: Task data (list of story, question, answer tuples).
    """
    with open(filepath) as f:
        return parse_stories(f.readlines())


def parse_stories(lines):
    """
    Parse stories in the bAbI Task format, return list of tuples of story, question,
    answers.

    :param lines: Lines of the task file.
    :return: List of story, question, answer tuples.
    """
    data, story = [], []
    for line in lines:
        line = line.lower()
        nid, line = line.split(' ', 1)
        nid = int(nid)

        # Check if new story
        if nid == 1:
            story = []

        # Check if question line
        if '\t' in line:
            q, a, _ = line.split('\t')
            q, a, substory = tokenize(q), [a], [x for x in story if x]

            # Remove question marks
            if q[-1] == "?":
                q = q[:-1]

            data.append((substory, q, a))
            story.append('')

        # Otherwise, it's just a story line
        else:
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def tokenize(sent):
    """
    Tokenize sentence, including punctuation.

    :param sent: Sentence to tokenize.
    :return: List of word tokens.
    """
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def vectorize_data(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories, queries, and answers.

    If sentence length < sentence_size, it will be padded with 0s.
    If story length < memory_size, stories will be padded with empty (zero) arrays.

    The answer array is encoded one-hot.

    :param data: Tuple of actual text (story, query, answer).
    :param word_idx: Vocabulary -> index dictionary
    :param sentence_size: Largest sentence size.
    :param memory_size: Size of memory.
    :return: Tuple of Story, Query, Answer NP Arrays
    """
    S, Q, A = [], [], []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # Take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # Pad to memory size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1)  # 0 is reserved for Null Word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)
