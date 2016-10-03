"""
lm.py

Core model definition file for the multitask language model. Given a list of languages, it produces
the necessary computation graphs for inference, loss calculation, and training.
"""
import tensorflow as tf


class LM:
    def __init__(self, languages, embedding_size, window_size, hidden_size, vocab_sizes,
                 learning_rate):
        """
        Initialize the Multi-Task Language Model, with the relevant parameters.

        :param languages: List of language identifiers to include.
        :param embedding_size: Size of internal language embeddings.
        :param window_size: Size of the fixed window.
        :param hidden_size: Size of the hidden shared layer (and joint embeddings).
        :param vocab_sizes: Dictionary mapping language identifier to vocabulary sizes.
        :param learning_rate: Learning Rate for Adam Optimizer
        """
        self.languages = languages
        self.embedding_size, self.window = embedding_size, window_size
        self.hidden, self.vocab_sizes = hidden_size, vocab_sizes
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False)

        # Setup Placeholders
        self.inputs, self.outputs = {}, {}
        for lang_id in self.languages:
            # Input is a Fixed Window of Words (referred to by idx in vocabulary)
            self.inputs[lang_id] = tf.placeholder(tf.int64, [None, self.window])  # bsz x window

            # Output is a single word (next word after window), referred to by idx in vocabulary
            self.outputs[lang_id] = tf.placeholder(tf.int64, [None])              # bsz

        # Instantiate all Model Weights
        self.instantiate_weights()

        # Build Inference Graphs
        self.logits, self.intermediate_embeddings = self.inference()

        # Build Loss Graphs
        self.loss_vals = self.loss()

        # Build Training Graphs
        self.train_ops = self.train()

    def instantiate_weights(self):
        """
        Initialize all the weights necessary for the model.
        """
        self.embeddings, self.encoders = {}, {}
        for lang_id in self.languages:
            self.embeddings[lang_id] = init_weight([self.vocab_sizes[lang_id], self.embedding_size],
                                                   "%s_embedding" % lang_id)
            self.encoders[lang_id] = (init_weight([self.window * self.embedding_size, self.hidden],
                                                  "%s_encoder" % lang_id),
                                      init_bias(self.hidden, 0.1, "%s_bias" % lang_id))

        self.hidden_relu, self.hidden_relu_bias = (init_weight([self.hidden, self.hidden],
                                                               "hidden_relu"),
                                                   init_bias(self.hidden, 0.1, "relu_bias"))
        self.hidden_linear, self.hidden_linear_bias = (init_weight([self.hidden, self.hidden],
                                                               "hidden_linear"),
                                                   init_bias(self.hidden, 0.1, "linear_bias"))

        self.decoders = {}
        for lang_id in self.languages:
            self.decoders[lang_id] = (init_weight([self.hidden, self.vocab_sizes[lang_id]],
                                                  "%s_decoder" % lang_id),
                                      init_bias(self.vocab_sizes[lang_id], 0.1, "%s_db" % lang_id))

    def inference(self):
        """
        Build the inference computation graph for the model.

        :return: Tuple of:
            Nested Dictionary mapping encoding language id -> decoding lang id -> logits.
            Dictionary mapping encoding language -> hidden embeddings
        """
        logit_paths, intermediate_embeddings = {}, {}
        for lang_id in self.languages:
            logit_paths[lang_id] = {}
            for lang_id2 in self.languages:
                logit_paths[lang_id][lang_id2] = None

        for lang_id in self.languages:
            # Get Embeddings for current Window
            window_input = self.inputs[lang_id]
            embedding = self.embeddings[lang_id]
            embeddings = tf.nn.embedding_lookup(embedding, window_input)     # bsz x window x embed

            # Flatten Embeddings into a Single Vector
            inter_sz = self.window * self.embedding_size
            flat = tf.reshape(embeddings, [-1, inter_sz])                    # bsz x (wdw x embed)

            # Pass Flattened Embeddings through Language Encoder
            encoder, encoder_bias = self.encoders[lang_id]
            encoding = tf.nn.relu(tf.matmul(flat, encoder) + encoder_bias)   # bsz x hidden

            # Pass Encodings Through First Shared ReLu Layer
            shared = tf.nn.relu(tf.matmul(encoding, self.hidden_relu) + self.hidden_relu_bias)

            # Pass Encodings Through Second Shared Linear Layer
            inter_embeddings = tf.matmul(shared, self.hidden_linear) + self.hidden_linear_bias
            intermediate_embeddings[lang_id] = inter_embeddings

        for decoding_language in self.languages:
            decoder, decoder_bias = self.decoders[decoding_language]
            for encoding_language in self.languages:
                # Pass Encoding through Decoder, get Output Logits
                encoding = intermediate_embeddings[encoding_language]
                decode_logits = tf.matmul(encoding, decoder) + decoder_bias
                logit_paths[encoding_language][decoding_language] = decode_logits

        return logit_paths, intermediate_embeddings

    def loss(self):
        """
        Build the loss computation graph for the model.

        :return Nested Dictionary mapping source lang -> target lang -> loss
        """
        losses = {}
        for src in self.languages:
            losses[src] = {}
            for trg in self.languages:
                logits = self.logits[src][trg]
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.outputs[trg],
                                                                   "%s_%s_loss" % (src, trg)))
                losses[src][trg] = loss
        return losses

    def train(self):
        """
        Build the training operations for the model.

        :return Nested Dictionary mapping source lang -> target lang -> train op
        """
        train_ops = {}
        opt = tf.train.AdamOptimizer(self.learning_rate)
        for src in self.languages:
            train_ops[src] = {}
            for trg in self.languages:
                train_ops[src][trg] = opt.minimize(self.loss_vals[src][trg],
                                                   global_step=self.global_step)
        return train_ops


def init_weight(shape, name):
    """
    Initialize a Tensor corresponding to a weight matrix with the given shape and name.

    :param shape: Shape of the weight tensor.
    :param name: Name of the weight tensor in the computation graph.
    :return: Tensor object with given shape and name, initialized from a standard normal.
    """
    return tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)


def init_bias(shape, value, name):
    """
    Initialize a Tensor corresponding to a bias vector with the given shape and name.

    :param shape: Shape of the bias vector (as an int, not a list).
    :param value: Value to initialize bias to.
    :param name: Name of the bias vector in the computation graph.
    :return: Tensor (Vector) object with given shape and name, initialized with given bias.
    """
    return tf.Variable(tf.constant(value, shape=[shape]), name=name)