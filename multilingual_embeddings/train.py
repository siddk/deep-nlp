"""
train.py

Core file for building and running the many-to-one multiple embedding model. Loads data,
preprocesses and tokenizes source files, then builds and trains the many-to-one model.
"""
from model.multitask_mt import MultiTaskMT
from preprocessor.reader import build_buckets, build_vocabularies, EOS_ID
from scipy.spatial.distance import cosine, euclidean
import math
import numpy as np
import os
import random
import sys
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("regularize", True, "Include self-self translation pairs.")
tf.app.flags.DEFINE_integer("max_sent_length", 35, "Maximum sentence length.")
tf.app.flags.DEFINE_string("raw_dir", "data/raw_data/", "Path to raw data directory.")
tf.app.flags.DEFINE_string("vocab_dir", "data/vocab/", "Path to vocabulary directory.")
tf.app.flags.DEFINE_string("bucket_dir", "data/buckets/", "Path to bucketed data directory.")
tf.app.flags.DEFINE_string("eval_dir", "data/eval/", "Path to eval results directory.")
tf.app.flags.DEFINE_string("log_dir", "log/", "Training directory.")

tf.app.flags.DEFINE_integer("hidden", 512, "Size of the internal model layers.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of stacked LSTM layers.")
tf.app.flags.DEFINE_integer("eval_every", 10, "Number of cycles of training to run before eval.")
tf.app.flags.DEFINE_integer("check_every", 200, "Number of steps to run before checkpoint.")

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")

# General Program Parameters
buckets = [(15, 15), (25, 25), (35, 35)]
source_languages = ["de", "es", "fr"]
target_languages = ["en"]
max_vocab_size = {"de": 50000, "fr": 50000, "en": 50000, "es": 50000}
order = ["en-es", "en-fr", "en-de", "en-en", "es-en", "es-es", "fr-en", "fr-fr", "de-en", "de-de"]
bucket_order = range(3)


# def load_data(bucket):
#     """
#     Load all parallel data from the corresponding bucket, for all languages.
#
#     :param bucket: Id of bucket to load from.
#     :return Return dictionary mapping language pairs to actual data.
#     """
#     data_dir = os.path.join(FLAGS.bucket_dir, str(bucket))
#     data = {k: [] for k in order}
#
#     for s in source_languages:
#         for t in target_languages:
#             prefix = '%s-%s' % tuple(sorted([s, t]))
#             with tf.gfile.GFile(os.path.join(data_dir, prefix + '.%s' % s)) as src:
#                 with tf.gfile.GFile(os.path.join(data_dir, prefix + ".%s" % t)) as trg:
#                     source, target = src.readline(), trg.readline()
#                     while source and target:
#                         source_ids = [int(x) for x in source.split()]
#                         target_ids = [int(x) for x in target.split()]
#                         data["%s-%s" % (s, t)].append([source_ids, target_ids + [EOS_ID]])
#                         data["%s-%s" % (t, s)].append([target_ids, source_ids + [EOS_ID]])
#                         data["%s-%s" % (s, s)].append([source_ids, source_ids + [EOS_ID]])
#                         data["%s-%s" % (t, t)].append([target_ids, target_ids + [EOS_ID]])
#                         source, target = src.readline(), trg.readline()
#     return data

def load_data(pair, bucket):
    """
    Load all parallel data from the corresponding bucket, for all languages.

    :param bucket: Id of bucket to load from.
    :return Return dictionary mapping language pairs to actual data.
    """
    data_dir = os.path.join(FLAGS.bucket_dir, str(bucket))
    data = {k: [] for k in order}

    s, t = pair.split('-')
    if not s == t:
        prefix = '%s-%s' % tuple(sorted([s, t]))
        with tf.gfile.GFile(os.path.join(data_dir, prefix + '.%s' % s)) as src:
            with tf.gfile.GFile(os.path.join(data_dir, prefix + ".%s" % t)) as trg:
                source, target = src.readline(), trg.readline()
                while source and target:
                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    data["%s-%s" % (s, t)].append([source_ids, target_ids + [EOS_ID]])
                    data["%s-%s" % (t, s)].append([target_ids, source_ids + [EOS_ID]])
                    data["%s-%s" % (s, s)].append([source_ids, source_ids + [EOS_ID]])
                    data["%s-%s" % (t, t)].append([target_ids, target_ids + [EOS_ID]])
                    source, target = src.readline(), trg.readline()
    else:
        files = [x for x in os.listdir(data_dir) if x[-2:] == s]
        target = np.random.choice(files)
        with tf.gfile.GFile(os.path.join(data_dir, target)) as src:
            source = src.readline()
            while source:
                source_ids = [int(x) for x in source.split()]
                data["%s-%s" % (s, s)].append([source_ids, source_ids + [EOS_ID]])
                source = src.readline()
    return data


def progress(count, total, suffix=''):
    """
    Simple progress bar function.

    :param count: Current count.
    :param total: Total possible count.
    :param suffix: String text to append to end of progress bar.
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


def get_distance_metrics(source_embeddings, target_embeddings):
    """
    Compute average cosine and euclidean distance for two sets of embeddings.
    """
    cosine_avg, euclidean_avg = 0.0, 0.0
    for i in range(len(source_embeddings)):
        cosine_avg += cosine(source_embeddings[i], target_embeddings[i])
        euclidean_avg += euclidean(source_embeddings[i], target_embeddings[i])
    return (cosine_avg / len(source_embeddings)), (euclidean_avg / len(source_embeddings))


def main(_):
    """
    Main function, loads and preprocesses data, then builds and trains the model.
    """
    build_vocabularies(source_languages, target_languages)
    build_buckets(buckets, max_vocab_size, source_languages, target_languages)

    # Load bucketed data (all in memory)
    # print "Loading data into memory (all buckets)"
    # data_set = []
    # for bucket_id in range(len(buckets)):
    #     data_set.append(load_data(bucket_id))

    # Start session
    with tf.Session() as sess:
        # Build model
        model = MultiTaskMT(source_languages, target_languages, max_vocab_size, FLAGS.hidden,
                            FLAGS.num_layers, buckets, FLAGS.learning_rate,
                            FLAGS.learning_rate_decay_factor, FLAGS.max_gradient_norm,
                            FLAGS.batch_size)
        ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
        data_set, prev_b = None, None
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print "Reading model parameters from %s. Restoring vars!" % ckpt.model_checkpoint_path
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "Created model with fresh parameters. Initializing vars!"
            sess.run(tf.initialize_all_variables())

        # Start Training Loop
        writer = tf.train.SummaryWriter("log/graph", sess.graph)
        print "Entering Training Loop"
        step_time, loss = 0.0, 0.0
        while True:
            # Set training index variables
            current_step = sess.run(model.global_step)
            previous_losses = []

            # Pick a bucket for training run
            b = np.random.choice(3)

            # Load data set into memory
            # if prev_b != b:
            #     data_set = load_data(b)
            # prev_b = b

            # Initialize cycle counter
            cycle_count = int(current_step) / int(len(order))

            # Evaluate model every so often.
            if cycle_count % FLAGS.eval_every == 0:
                eval_data = [('de-en-eval.de', 'de-en-eval.en'), ('en-es-eval.en', 'en-es-eval.es'),
                             ('en-fr-eval.en', 'en-fr-eval.fr')]
                for b in range(len(buckets)):
                    for s, t in eval_data:
                        source_data, target_data = [], []
                        path = os.path.join(FLAGS.bucket_dir, str(b))
                        with tf.gfile.GFile(os.path.join(path, s)) as src:
                            with tf.gfile.GFile(os.path.join(path, t)) as trg:
                                source, target = src.readline(), trg.readline()
                                while source and target:
                                    source_data.append([int(x) for x in source.split()])
                                    target_data.append([int(x) for x in target.split()])
                                    source, target = src.readline(), trg.readline()
                        src_embeddings, trg_embeddings = model.eval(sess, source_data, target_data,
                                                                    s[-2:], t[-2:], b)
                        random.shuffle(target_data)
                        shuf_src, shuf_trg = model.eval(sess, source_data, target_data, s[-2:],
                                                        t[-2:], b)

                        cos_dist, euc_dist = get_distance_metrics(src_embeddings, trg_embeddings)
                        shuf_cos, shuf_euc = get_distance_metrics(shuf_src, shuf_trg)

                        print "Average Cosine Distance for %s-%s in bucket %s is %s" % (s[-2:],
                                                                                        t[-2:], b,
                                                                                        cos_dist)
                        print "Average Euclidean Distance for %s-%s in bucket %s is %s" % (s[-2:],
                                                                                           t[-2:],
                                                                                           b,
                                                                                           euc_dist)
                        print "Average Cosine Distance for Random %s-%s in bucket %s is %s" % (
                            s[-2:], t[-2:], b, shuf_cos
                        )
                        print "Average Euclidean Distance for Random %s-%s in bucket %s is %s" % (
                            s[-2:], t[-2:], b, shuf_euc
                        )
                        print "Cosine Distance between Random and Real is the same: %s" % (
                            str(shuf_cos == cos_dist)
                        )
                        print "Euclidean Distance between Random and Real is the same: %s" % (
                            str(shuf_euc == euc_dist)
                        )
                        with open(os.path.join(FLAGS.eval_dir, '%s_%s-%s.cos' % (str(b), s[-2:], t[-2:])), 'a+') as f:
                            f.write("%s\t%s\n" % (str(cycle_count), str(cos_dist)))

                        with open(os.path.join(FLAGS.eval_dir, '%s_%s-%s.euc' % (str(b), s[-2:], t[-2:])), 'a+') as f:
                            f.write("%s\t%s\n" % (str(cycle_count), str(euc_dist)))

                        with open(os.path.join(FLAGS.eval_dir, 'random_%s_%s-%s.cos' % (str(b), s[-2:], t[-2:])), 'a+') as f:
                            f.write("%s\t%s\n" % (str(cycle_count), str(shuf_cos)))

                        with open(os.path.join(FLAGS.eval_dir, 'random_%s_%s-%s.euc' % (str(b), s[-2:], t[-2:])), 'a+') as f:
                            f.write("%s\t%s\n" % (str(cycle_count), str(shuf_euc)))

            # Iterate through order cycle
            for p in order:
                print "Loading data for", p
                data_set = load_data(p, b)
                # Get a batch and make a step.
                start_time = time.time()
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(data_set[p], b)
                step_loss, embeddings, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                      target_weights, p, b, False)
                step_time += (time.time() - start_time) / FLAGS.check_every
                loss += step_loss / FLAGS.check_every
                current_step += 1
                progress(current_step % FLAGS.check_every, FLAGS.check_every,
                         "  Step %s" % (current_step / FLAGS.check_every))

                # Once in a while, we save checkpoint, and print statistics.
                if current_step % FLAGS.check_every == 0:
                    # Print statistics for the previous chunk.
                    print ""
                    perplexity = math.exp(loss) if loss < 1000 else float('inf')
                    print ("Global step %d, Learning rate %.4f, Step-time %.2f, Perplexity %.2f" %
                          (model.global_step.eval(), model.learning_rate.eval(), step_time,
                           perplexity))

                    with open(os.path.join(FLAGS.eval_dir, 'loss.txt'), 'a+') as loss_file:
                        loss_file.write("%s    %s\n" % (str(model.global_step.eval()), str(perplexity)))

                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        sess.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)

                    # Save checkpoint and zero timer and loss.
                    checkpoint_path = os.path.join(FLAGS.log_dir, "translate.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    step_time, loss = 0.0, 0.0
                    sys.stdout.flush()
                    print ""
            cycle_count += 1





if __name__ == "__main__":
    tf.app.run()