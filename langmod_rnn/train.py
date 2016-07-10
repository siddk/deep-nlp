"""
train.py

Core file for building, training, and evaluating the RNN (LSTM) Language Model on the Hansard's
Data Set. Builds a three-layer Recurrent Neural Net, implicitly learning long term word
dependencies.
"""
from model.langmod_rnn import LangmodLSTM
from preprocessor.reader import get_examples
import os
import tensorflow as tf

# Set up Tensorflow Training Parameters, or FLAGS
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_path", "data/english-senate-0.txt", "Path to training data.")
tf.app.flags.DEFINE_string("test_path", "data/english-senate-2.txt", "Path to test data.")
tf.app.flags.DEFINE_string('log_dir', 'log/checkpoints', "Directory where to write checkpoints.")
tf.app.flags.DEFINE_string('summary_dir', 'log/summaries', 'Directory where to write summaries.')

# Model Parameters
tf.app.flags.DEFINE_integer('vocab_size', 15000, 'Size of the vocabulary.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size, for training.')
tf.app.flags.DEFINE_integer('window_size', 20, 'Size of each window.')
tf.app.flags.DEFINE_integer('embedding_size', 20, 'Dimension of the embeddings.')
tf.app.flags.DEFINE_integer('hidden_size', 100, 'Size of the hidden (LSTM) layers.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of stacked LSTM layers.')

# Training parameters
tf.app.flags.DEFINE_float('max_grad_norm', 5, "Maximum gradient norm to clip to.")
tf.app.flags.DEFINE_integer('epochs', 10, 'Number of training epochs.')


def main(_):
    """
    Main function, reads and parses data, builds and trains model, then evaluates on the test
    data.
    """
    # Fix directories
    if tf.gfile.Exists(FLAGS.summary_dir):
        tf.gfile.DeleteRecursively(FLAGS.summary_dir)
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.summary_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Load Data
    train_x, train_y, vocab = get_examples(FLAGS.train_path)
    test_x, test_y, _ = get_examples(FLAGS.test_path, vocab)

    # Start Tensorflow Session
    with tf.Session() as sess:
        print "Starting Training!"
        # Instantiate Network
        langmod_lstm = LangmodLSTM()

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())
        num_batches, bsz = train_x.shape[0], FLAGS.batch_size

        # Start training
        test_l, train_l = [], []
        for epoch in range(FLAGS.epochs):
            lr_decay = 0.5 ** max(epoch - 5, 0.0)
            langmod_lstm.assign_lr(sess, 1.0 * lr_decay)
            epoch_loss = 0.0
            state = langmod_lstm.initial_state.eval()
            for i in range(num_batches):
                loss, state, _ = sess.run([langmod_lstm.loss_val, langmod_lstm.final_state,
                                          langmod_lstm.train_op],
                                          feed_dict={langmod_lstm.X: train_x[i],
                                                     langmod_lstm.Y: train_y[i],
                                                     langmod_lstm.initial_state: state})
                epoch_loss += loss
                if i % 2 == 0:
                    print "Batch", i, "Loss:", loss

            print "Training Loss for Epoch", epoch, "is:", epoch_loss
            train_l.append(epoch_loss)

            # Evaluate Test Loss
            test_loss = 0.0
            state = langmod_lstm.initial_state.eval()
            for i in range(test_x.shape[0]):
                loss, state = sess.run([langmod_lstm.loss_val, langmod_lstm.final_state],
                                       feed_dict={langmod_lstm.X: test_x[i],
                                                  langmod_lstm.Y: test_y[i],
                                                  langmod_lstm.initial_state: state})
                test_loss += loss

            print "Test Loss for Epoch", epoch, "is:", test_loss
            test_l.append(test_loss)

            # Save model
            checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, epoch)

        print "Training Losses", train_l
        print "Test Losses", test_l


if __name__ == "__main__":
    tf.app.run()