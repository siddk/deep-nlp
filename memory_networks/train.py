"""
train.py

Core file for training an End-to-End Memory Network for Facebook's bAbI Task Dataset. Given a
story and a series of questions about the story, the network tries to predict the answer to the
question using a memory network to build an attention mechanism over sentences in the story, and
pick out the relevant parts.
"""
from itertools import chain
from model.memn2n import MemN2N
from preprocessor.reader import load_task, vectorize_data
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
import tensorflow as tf
import sys

# Set up Tensorflow Training Parameters, or Flags
FLAGS = tf.flags.FLAGS

# Data Parameters
tf.flags.DEFINE_integer("task_id", 2, "bAbI Task (1 - 20). See bAbI Task README for more info.")
tf.flags.DEFINE_string("data_dir", "data/en", "Path to 1k Example Task Data. For 10k, use en-10k.")
tf.app.flags.DEFINE_string('log_dir', 'log/checkpoints/task2', "Directory to write checkpoints.")

# Model Parameters
tf.flags.DEFINE_integer("memory_size", 50, "Maximum number of sentences in memory.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")

# Train Parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("linear_start_epochs", 50, "Number of linear epochs.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")


def main(_):
    """
    Main training function, loads and vectorizes the data for the given Task, then
    proceeds to train and evaluate the model.
    """
    # Fix directories
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Load Task Data
    train, test = load_task(FLAGS.data_dir, FLAGS.task_id)

    # Build vocabulary
    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a)
                                               for s, q, a in train + test)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    # Calculate maximum size parameters
    max_story_size = max(map(len, (s for s, _, _ in train + test)))
    mean_story_size = int(np.mean(map(len, (s for s, _, _ in train + test))))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in train + test)))
    query_size = max(map(len, (q for _, q, _ in train + test)))
    memory_size = min(FLAGS.memory_size, max_story_size)
    vocab_size = len(word_idx) + 1                  # +1 for null word
    sentence_size = max(query_size, sentence_size)  # For the position

    print "Longest sentence length", sentence_size
    print "Longest story length", max_story_size
    print "Average story length", mean_story_size

    # Vectorize training and test data
    S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size)
    trainS, valS, trainQ, valQ, trainA, valA = train_test_split(S, Q, A, test_size=.1)
    testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

    train_labels = np.argmax(trainA, axis=1)
    test_labels = np.argmax(testA, axis=1)
    val_labels = np.argmax(valA, axis=1)

    n_train, batch_size = trainS.shape[0], FLAGS.batch_size

    # Start Session
    print "Starting Training"
    with tf.Session() as sess:
        # Instantiate Network
        memn2n = MemN2N(vocab_size, sentence_size, memory_size)

        # Create a Saver
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())
        print "Variables Initialized!"
        print ""

        # Run through the linear start epochs
        for t in range(1, FLAGS.linear_start_epochs + 1):
            print "Linear Start Epoch:", t
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                s, q, a = trainS[start:end], trainQ[start:end], trainA[start:end]
                _, _ = sess.run([memn2n.linear_loss_val, memn2n.linear_train_op],
                                feed_dict={memn2n.stories: s, memn2n.questions: q,
                                           memn2n.answers: a})

        # Run through the epochs (for real)
        for t in range(1, FLAGS.epochs + 1):
            total_cost = 0.
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                s, q, a = trainS[start:end], trainQ[start:end], trainA[start:end]
                batch_loss, _ = sess.run([memn2n.loss_val, memn2n.train_op],
                                         feed_dict={memn2n.stories: s, memn2n.questions: q,
                                                    memn2n.answers: a})
                total_cost += batch_loss

            if t % FLAGS.evaluation_interval == 0:
                train_preds = []
                for start in range(0, n_train, batch_size):
                    end = start + batch_size
                    s, q = trainS[start:end], trainQ[start:end]
                    pred = sess.run(memn2n.predict_op, feed_dict={memn2n.stories: s,
                                                                  memn2n.questions: q})
                    train_preds += list(pred)

                val_preds = sess.run(memn2n.predict_op, feed_dict={memn2n.stories: valS,
                                                                   memn2n.questions: valQ})
                train_acc = accuracy_score(np.array(train_preds), train_labels)
                val_acc = accuracy_score(val_preds, val_labels)

                print '-----------------------'
                print 'Epoch', t
                print 'Total Cost:', total_cost
                print 'Training Accuracy:', train_acc
                print 'Validation Accuracy:', val_acc
                print '-----------------------'
                print ''

                # Save the model
                checkpoint_path = os.path.join(FLAGS.log_dir, 'model_task_' + str(FLAGS.task_id)
                                                                            + '.ckpt')
                saver.save(sess, checkpoint_path, t)

                # Flush stdout
                sys.stdout.flush()

        # Evaluate on Test
        test_preds = sess.run(memn2n.predict_op, feed_dict={memn2n.stories: testS,
                                                            memn2n.questions:testQ})
        test_acc = accuracy_score(test_preds, test_labels)
        print "Testing Accuracy:", test_acc


if __name__ == "__main__":
    tf.app.run()