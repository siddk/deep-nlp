"""
train.py

Core file for Encoder-Decoder Machine Translation based on Cho et. al. 2014. Builds an LSTM encoder
which produces a state vector that gets fed into a separate LSTM decoder, to perform machine
translation. Translates from French to English, using the Canadian Hansard's Dataset.
"""
from model.seq2seq import Seq2Seq
from preprocessor.reader import load_data, init_vocab
import math
import numpy as np
import os
import preprocessor
import sys
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", "data/", "Data directory")
tf.app.flags.DEFINE_string("log_dir", "log_intermediate/", "Training directory.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps per checkpoint.")

tf.app.flags.DEFINE_integer("max_vsz", 40000, "Maximum size of a single language vocabulary.")
tf.app.flags.DEFINE_integer("max_size", 50, "Maximum size of sentence.")

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")

buckets = [(10, 5), (15, 10), (25, 20), (35, 30), (50, 50)]


def bucket(source_path, target_path):
    """
    Given path to source and target token files, read all the data, and place them into the
    appropriate buckets.

    :param source_path: Path to source tokens
    :param target_path: Path to target tokens
    """
    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(source_path, 'r') as source_file:
        with tf.gfile.GFile(target_path, 'r') as target_file:
            source, target = source_file.readline(), target_file.readline()
            while source and target:
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(preprocessor.reader.EOS_ID)

                # Find a bucket
                for bucket_id, (source_size, target_size) in enumerate(buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break

                source, target = source_file.readline(), target_file.readline()
    return data_set


def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


def main(_):
    """
    Main function, loads and vectorizes data, builds model, then proceeds to start the training
    process.
    """
    # Load data (paths to the vocab, and tokens)
    fr_vocab, en_vocab, fr_train, en_train, _ = load_data()

    # Bucket train data
    train_set = bucket(fr_train, en_train)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    print "Total Number of Training Examples", train_total_size

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # Get size of vocabularies
    french_vocab, _ = init_vocab(fr_vocab)
    english_vocab, _ = init_vocab(en_vocab)

    # Start Tensorflow Session
    with tf.Session() as sess:
        model = Seq2Seq(len(french_vocab), len(english_vocab), buckets, FLAGS.size,
                        FLAGS.num_layers, forward_only=False)
        ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print "Reading model parameters from %s" % ckpt.model_checkpoint_path
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "Created model with fresh parameters."
            sess.run(tf.initialize_all_variables())

        # Start Training Loop
        step_time, loss, current_step = 0.0, 0.0, 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number in [0, 1] and
            # use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            step_loss, embeddings, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                  target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1
            progress(current_step % FLAGS.steps_per_checkpoint, FLAGS.steps_per_checkpoint,
                     "Step %s" % (current_step / FLAGS.steps_per_checkpoint))

            # Once in a while, we save checkpoint, and print statistics.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                print ""
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("Global step %d, Learning rate %.4f, Step-time %.2f, Perplexity %.2f" %
                      (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))

                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)

                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.log_dir, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

if __name__ == "__main__":
    tf.app.run()

