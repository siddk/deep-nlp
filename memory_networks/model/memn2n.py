"""
memn2n.py

Core implementation for End-to-End Memory Networks as detailed in the original paper, here:
http://arxiv.org/abs/1503.08895. Builds a Multiple Hop Memory Network with
adjacent weight Tying and Position Encoding for Facebook's bAbI Task Question-Answering dataset.
"""
import numpy as np
import tensorflow as tf

# Get Tensorflow Parameters
FLAGS = tf.flags.FLAGS


class MemN2N:
    def __init__(self, vocab_size, sentence_size, memory_size, name="MemN2N"):
        """
        Initialize an End-to-End Memory Network, with the necessary hyperparameters.

        :param vocab_size: The size of the vocabulary (including the null word).
        :param sentence_size: The max size of a sentence in the data. All sentences should be padded
                              to this length.
        :param memory_size: The max size of the memory. Since Tensorflow currently does not support
                            jagged arrays all memories must be padded to this length.
        """
        self.name = name
        self.vocab_size = vocab_size
        self.sentence_size = sentence_size
        self.memory_size = memory_size
        self.embedding_size = FLAGS.embedding_size
        self.hops = FLAGS.hops
        self.init = tf.random_normal_initializer(stddev=0.1)
        self.encoding = tf.constant(position_encoding(self.sentence_size, self.embedding_size),
                                    name="encoding")

        # Setup Placeholders
        self.stories = tf.placeholder(tf.int32, [None, self.memory_size, self.sentence_size],
                                      name='stories')
        self.questions = tf.placeholder(tf.int32, [None, self.sentence_size], name='questions')
        self.answers = tf.placeholder(tf.int32, [None, self.vocab_size], name='answers')

        # Instantiate Weights
        self.instantiate_weights()

        # Build the model, return the logits
        self.linear_logits = self.inference(True)
        self.logits = self.inference(False)

        # Build the loss computation graph, using the logits
        self.linear_loss_val = self.loss(True)
        self.loss_val = self.loss(False)

        # Set up the training_operation
        self.linear_train_op = self.train(True)
        self.train_op = self.train(False)

        # Set up prediction operations
        self.predict_op = tf.argmax(self.logits, 1, name="predict_op")

    def instantiate_weights(self):
        """
        Create weight variables (all model trainable parameters).
        """
        with tf.variable_scope(self.name):
            self.nil_vars = []
            # Setup Embedding Matrices (Zero out null word).
            null_word_slot = tf.zeros([1, self.embedding_size])

            # Layer 0 Question Embedding
            self.B = tf.Variable(tf.concat(0,
                                           [null_word_slot,
                                            self.init([self.vocab_size - 1, self.embedding_size])]),
                                 name="B")
            self.nil_vars.append("B")

            # Adjacent Tied Story Embeddings (last one is shared with final tensor W)
            self.story_embeddings = []
            for i in range(self.hops):
                self.story_embeddings.append(tf.Variable(tf.concat(0,
                                                                   [null_word_slot,
                                                                    self.init(
                                                                        [self.vocab_size - 1,
                                                                         self.embedding_size])]),
                                                         name="A" + str(i)))
                self.nil_vars.append("A" + str(i))

            # Adjacent Tied Temporal Encoding Matrices
            self.temporal_encodings = []
            for i in range(self.hops + 1):
                self.temporal_encodings.append(tf.Variable(self.init([self.memory_size,
                                                                      self.embedding_size]),
                                                           name='TA' + str(i)))

            # Between Layer Linear Transformation
            self.H = tf.Variable(self.init([self.embedding_size, self.embedding_size]), name="H")

            # Final Transformation (also shared with Embedding C) -> Needs to be transposed
            self.W = tf.Variable(tf.concat(0,
                                           [null_word_slot,
                                            self.init([self.vocab_size - 1, self.embedding_size])]),
                                 name="W")
            self.nil_vars.append("W")

    def inference(self, linear_start=True):
        """
        Build the core of the model, describing all transformations and connections between
        the inputs. Setup the entire computation graph.

        :return Tensor representing the network logits for the given inputs.
        """
        with tf.variable_scope(self.name):
            q_embedding = tf.nn.embedding_lookup(self.B, self.questions)
            u = [tf.reduce_sum(q_embedding * self.encoding, 1)]
            for i in range(self.hops):
                # Get current A Embedding, TA Encoding
                A = self.story_embeddings[i]
                TA = self.temporal_encodings[i]

                # Get current C Embedding, TC Encoding
                if i == self.hops - 1:
                    C = self.W
                else:
                    C = self.story_embeddings[i + 1]
                TC = self.temporal_encodings[i + 1]

                # Compute Memory Embedding A -> Position Encoding
                mem_embedding_a = tf.nn.embedding_lookup(A, self.stories)
                m_a = tf.reduce_sum(mem_embedding_a * self.encoding, 2) + TA

                # Hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m_a * u_temp, 2)

                # Calculate probabilities
                if linear_start:
                    probs = dotted
                else:
                    probs = tf.nn.softmax(dotted)
                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])

                # Compute Memory Embedding C -> Position Encoding
                mem_embedding_c = tf.nn.embedding_lookup(C, self.stories)
                m_c = tf.reduce_sum(mem_embedding_c * self.encoding, 2) + TC

                # Calculate attention
                c_temp = tf.transpose(m_c, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                # Compute input to next layer
                u.append(tf.matmul(u[-1], self.H) + o_k)

            return tf.matmul(u[-1], tf.transpose(self.W, [1, 0]))

    def loss(self, linear_start=True):
        """
        Compute the loss computation graph, using the logits.
        """
        if linear_start:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.linear_logits,
                                                                    tf.cast(self.answers, tf.float32),
                                                                    name="linear_cross_entropy")
        else:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits,
                                                                    tf.cast(self.answers, tf.float32),
                                                                    name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")
        return cross_entropy_sum

    def train(self, linear_start=True):
        """
        Compute the training operation computation graph.
        """
        self.opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)
        if linear_start:
            grads_and_vars = self.opt.compute_gradients(self.linear_loss_val)
        else:
            grads_and_vars = self.opt.compute_gradients(self.loss_val)
        grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v) for g, v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self.nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        if linear_start:
            train_op = self.opt.apply_gradients(nil_grads_and_vars, name="linear_train_op")
        else:
            train_op = self.opt.apply_gradients(nil_grads_and_vars, name="train_op")
        return train_op


def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.
    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.

    :param t: Tensor to overwrite
    """
    with tf.op_scope([t], name, "zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.pack([1, s]))
        return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks [2].

    :param t: Tensor to add noise to.
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)
