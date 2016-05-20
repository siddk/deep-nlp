"""
eval.py

Core runner for evaluating the Deep CIFAR-10 model. Loads model from training checkpoints, and
builds a new inference pipeline with the saved weights. Evaluates on the CIFAR Test set.

Credit: Tensorflow CIFAR Tutorial: http://tensorflow.org/tutorials/deep_cnn/
"""
from datetime import datetime
from model.cifar import CIFAR, MOVING_AVERAGE_DECAY
import math
import numpy as np
import tensorflow as tf
import time

# Tensorflow Evaluation Parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', 'tmp/cifar10_eval', "Directory where to write event logs.")
tf.app.flags.DEFINE_string('eval_data', 'test', "Either 'test' or 'train_eval'.")
tf.app.flags.DEFINE_string('checkpoint_dir', 'tmp/cifar_train',
                           "Directory where to read model checkpoints.")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5, "How often to run the eval.")
tf.app.flags.DEFINE_integer('num_examples', 10000, "Number of examples to run.")
tf.app.flags.DEFINE_boolean('run_once', True, "Whether to run eval only once.")
tf.app.flags.DEFINE_string('data_dir', 'tmp/cifar10_data', "Path to the CIFAR-10 data directory.")
tf.app.flags.DEFINE_integer('batch_size', 128, "Number of images to process in a batch.")

def evaluate():
    """
    Evaluate the CIFAR-10 Model on the test data.
    """
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        eval_data = FLAGS.eval_data == 'test'
        images, labels = CIFAR.inputs(eval_data=eval_data)

        # Instantiate a CIFAR Evaluation Model with inference, loss, and train_ops
        cifar = CIFAR(images, labels, None, train=False)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, cifar.top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """
    Run Eval once
    :param saver: Saver
    :param summary_writer: Summary writer
    :param top_k_op: Top K op
    :param summary_op: Summary op
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0

            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def main(argv=None):
    """
    Main Function, called by tf.app.run(). Downloads the Test Data if Necessary, then evaluates
    the saved model.
    """
    CIFAR.download_and_extract()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()

# Python Main function, starts the tensorflow session by invoking main() [see above]
if __name__ == "__main__":
    tf.app.run()