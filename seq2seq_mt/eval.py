"""
eval.py

Core file for evaluating (translating) with the seq2seq Machine Translation model. Given a series
of sentences (as a source file) in French, run through the model, and decode to English.
"""
import numpy as np
import sys
import tensorflow as tf
from model.seq2seq import Seq2Seq
from preprocessor.reader import init_vocab, EOS_ID

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", "data/", "Data directory")
tf.app.flags.DEFINE_string("log_dir", "log_intermediate/", "Training directory.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps per checkpoint.")

tf.app.flags.DEFINE_integer("max_vsz", 20000, "Maximum size of a single language vocabulary.")
tf.app.flags.DEFINE_integer("max_size", 50, "Maximum size of sentence.")

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")

SOURCE_PATH = "data/tokens/fr.train"
TARGET_PATH = "data/out_intermediate.txt"
buckets = [(10, 5), (15, 10), (25, 20), (35, 30), (50, 50)]


def main(_):
    """
    Main function, instantiates model, loads and vectorizes source data, translates and outputs
    English translations.
    """
    # Load vocabularies
    fr_vocab_path = "data/vocabulary/fr.vocab"
    en_vocab_path = "data/vocabulary/en.vocab"

    fr2idx, idx2fr = init_vocab(fr_vocab_path)
    en2idx, idx2en = init_vocab(en_vocab_path)

    with tf.Session() as sess:
        # Create Model by Loading Parameters
        model = Seq2Seq(len(fr2idx), len(en2idx), buckets, FLAGS.size,
                        FLAGS.num_layers, forward_only=True)
        ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print "Reading model parameters from %s" % ckpt.model_checkpoint_path
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "No model checkpoints found!"
            sys.exit(0)

        # Reset batch_size to 1
        model.batch_size = 1
        translations = []
        with tf.gfile.GFile(SOURCE_PATH, 'rb') as f:
            sentence = f.readline()
            while sentence:
                # Source file is already tokenized, just need to split at spaces
                token_ids = sentence.split()
                if len(token_ids) >= 50:
                    translations.append("")
                    sentence = f.readline()
                    continue

                # Pick which bucket it belongs to.
                bucket_id = min([b for b in xrange(len(buckets)) if buckets[b][0] > len(token_ids)])

                # Get a 1-element batch to feed the sentence to the model.
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    {bucket_id: [(token_ids, [])]}, bucket_id)

                # Get output logits for the sentence.
                _, embedding, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                              target_weights, bucket_id, True)

                # This is a greedy decoder - outputs are just argmaxes of output_logits.
                outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

                # If there is an EOS symbol in outputs, cut them at that point.
                if EOS_ID in outputs:
                    outputs = outputs[:outputs.index(EOS_ID)]

                # Print out English sentence corresponding to outputs.
                translation = " ".join([tf.compat.as_str(idx2en[output]) for output in outputs])
                print translation
                translations.append(translation)
                sentence = f.readline()

        with tf.gfile.GFile(TARGET_PATH, 'wb') as f:
            for t in translations:
                f.write(t + "\n")

if __name__ == "__main__":
    tf.app.run()