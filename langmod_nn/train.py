"""
train.py

Core runner for training the simple Deep Bigram Language Model. Builds a three-layer neural net,
consisting of an embedding layer, a ReLU layer, and a softmax layer.
"""
from model.langmod import Langmod
from preprocessor.reader import *
import tensorflow as tf
import time
import os

# Set up Tensorflow Training Parameters, or "FLAGS"
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir', 'log/checkpoints', "Directory where to write checkpoints.")
tf.app.flags.DEFINE_string('summary_dir', 'log/summaries', 'Directory where to write summaries.')
tf.app.flags.DEFINE_integer('epochs', 6, "Maximum number of epochs to run.")
tf.app.flags.DEFINE_integer('batch_size', 128, "Number of bigrams to process in a batch.")
tf.app.flags.DEFINE_float('learning_rate', 0.05, "Learning rate for SGD.")
tf.app.flags.DEFINE_integer('embedding_size', 50, "Dimension of the embeddings.")
tf.app.flags.DEFINE_integer('hidden_size', 100, "Dimension of the hidden layer.")
tf.app.flags.DEFINE_string('train_path', 'data/english-senate-0.txt', "Path to train data.")
tf.app.flags.DEFINE_string('test_path', 'data/english-senate-2.txt', "Path to test data.")
tf.app.flags.DEFINE_integer('vocab_size', 5000, "Size of the vocabulary.")


def main(_):
    """
    Main Training function, loads and preprocesses training data, then builds the model.
    """
    # Fix directories
    if tf.gfile.Exists(FLAGS.summary_dir):
        tf.gfile.DeleteRecursively(FLAGS.summary_dir)
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.summary_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Load Data
    x, y, vocab = get_bigrams(FLAGS.train_path)
    test_x, test_y, _ = get_bigrams(FLAGS.test_path, vocab)

    with tf.Session() as sess:
        # Instantiate Network
        langmod = Langmod()

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build summary operations
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.summary_dir + "/train", sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summary_dir + "/test")

        # Initialize all variables
        sess.run(tf.initialize_all_variables())
        bsz = FLAGS.batch_size

        # Start training
        for epoch in range(FLAGS.epochs):
            # Split up batches
            print ""
            counter = 0
            start_time = time.time()
            for start, end in zip(range(0, len(x), bsz), range(bsz, len(x), bsz)):
                counter += 1
                _ = sess.run([langmod.train_op], feed_dict={langmod.X: x[start:end],
                                                            langmod.Y: y[start:end]})
                if counter % 100 == 0:
                    print "Processing batch", counter, "of epoch", epoch

            # Evaluate Training loss
            tr_summary, tr_cost, tr_ll = sess.run([merged, langmod.loss_val, langmod.log_lik],
                                                  feed_dict={langmod.X: x, langmod.Y: y})
            train_writer.add_summary(tr_summary, epoch)
            print ""
            print "Training loss after epoch", epoch, "is: ", tr_cost
            print "Training Log Likelihood after epoch", epoch, "is: ", tr_ll

            # Evaluate Test Loss
            tst_summary, tst_cost, tst_ll = sess.run([merged, langmod.loss_val, langmod.log_lik],
                                                     feed_dict={langmod.X: test_x,langmod.Y: test_y})

            test_writer.add_summary(tst_summary, epoch)
            print ""
            print "Test loss after epoch", epoch, "is: ", tst_cost
            print "Test Log Likelihood after epoch", epoch, "is: ", tst_ll

            # Save model
            checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, bsz * epoch)
            print "Epoch", epoch, "took: ", time.time() - start_time, "seconds!"
            print ""


if __name__ == "__main__":
    tf.app.run()