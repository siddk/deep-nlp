"""
seq2seq.py

Core model file for the Encoder-Decoder Machine Translation system, based off of Cho et. al. Builds
a bucketed Encoder-Decoder model, to translate from French to English.
"""
from preprocessor import reader
import numpy as np
import random
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


class Seq2Seq:
    def __init__(self, source_vsz, target_vsz, buckets, hidden_size, num_layers, num_samples=512,
                 forward_only=False):
        """
        Initialize a Sequence-to-Sequence Encoder-Decoder model, for Machine Translation.

        :param source_vsz: Size of the source vocabulary
        :param target_vsz: Size of the target vocabulary
        :param buckets: List of pairs of each bucket size
        :param hidden_size: Number of units in each of the Hidden Layers
        :param num_layers: Number of encoder/decoder LSTM layers
        :param num_samples: Number of samples for sampled softmax
        """
        self.source_vsz, self.target_vsz = source_vsz, target_vsz
        self.buckets = buckets
        self.hidden, self.num_layers = hidden_size, num_layers
        self.global_step = tf.Variable(0, trainable=False)
        self.max_gradient_norm = FLAGS.max_gradient_norm
        self.batch_size = FLAGS.batch_size
        self.learning_rate = tf.Variable(float(FLAGS.learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate *
                                                                FLAGS.learning_rate_decay_factor)

        # Set up sampled softmax parameters
        self.output_proj, self.sampled_softmax_func = self.sampled_softmax(num_samples, target_vsz)

        # Set up placeholders -> Encoder/Decoder takes a list of inputs, not a tensor
        self.encoder_inputs, self.decoder_inputs = [], []
        self.target_weights = []

        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{0}".format(i)))

        # Targets are decoder inputs shifted by one.
        self.targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

        # Instantiate necessary weights and RNN Cells
        self.instantiate_weights()

        # Build inference pipeline
        if forward_only:
            self.outputs, self.losses = self.inference(True)
            if self.output_proj is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [tf.matmul(output, self.output_proj[0]) + self.output_proj[1]
                                       for output in self.outputs[b]]
        else:
            self.outputs, self.losses = self.inference(False)

        # Set up training graph
        self.train()

        # Set up saver
        self.saver = tf.train.Saver(tf.all_variables())

    def sampled_softmax(self, num_samples, target_vsz):
        """
        Setup the output projection parameters for the calculation of the sampled softmax loss
        function.

        :param num_samples: Number of samples for sampled softmax
        :param target_vsz: Size of target vocabulary
        :return: Tuple of the output projection (w, b), and the actual softmax loss function op
        """
        self.w = tf.get_variable("proj_w", [self.hidden, self.target_vsz])
        self.w_t = w_t = tf.transpose(self.w)
        self.b = b = tf.get_variable("proj_b", [self.target_vsz])

        def sampled_loss(inputs, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, target_vsz)

        return (self.w, self.b), sampled_loss

    def instantiate_weights(self):
        """
        Instantiate network weights and RNN Cells, for use in the inference process.
        """
        self.single_cell = tf.nn.rnn_cell.GRUCell(self.hidden)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([self.single_cell] * self.num_layers)

        self.inter_weight = tf.get_variable("inter_w", [self.num_layers * self.hidden,
                                                        self.num_layers * self.hidden])
        self.inter_bias = tf.get_variable("inter_b", [self.num_layers * self.hidden])

        self.embed_weight = tf.get_variable("embed_w", [self.num_layers * self.hidden,
                                                        self.num_layers * self.hidden])
        self.embed_bias = tf.get_variable("embed_b", [self.num_layers * self.hidden])

    def inference(self, do_decode):
        """
        Build core inference computation graph, mapping the inputs to the outputs.

        :param do_decode: Boolean if to feed previous output, or use true output in decode step.
        :return: Tuple of RNN Outputs, and RNN Losses
        """
        # Setup placeholder for stealing the intermediate state, for use later.
        self.intermediate_embeddings = []
        outputs, losses = [], []

        # Bucket
        for j, bucket in enumerate(self.buckets):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 else None):
                encoder_cell = tf.nn.rnn_cell.EmbeddingWrapper(self.cell,
                                                               embedding_classes=self.source_vsz,
                                                               embedding_size=self.hidden)
                _, encoder_state = tf.nn.rnn(encoder_cell, self.encoder_inputs[:bucket[0]],
                                             dtype=tf.float32)

                # Transform the encoder state through a ReLU Hidden Layer
                intermediate_state = tf.matmul(encoder_state, self.inter_weight) + self.inter_bias
                intermediate_relu = tf.nn.relu(intermediate_state, "inter_relu")

                # Linear transform, get and append intermediate embedding
                embedding = tf.matmul(intermediate_relu, self.embed_weight) + self.embed_bias
                self.intermediate_embeddings.append(embedding)

                # Get decoder outputs
                bucket_outputs, _ = tf.nn.seq2seq.embedding_rnn_decoder(
                    self.decoder_inputs[:bucket[1]], embedding, self.cell, self.target_vsz,
                    self.hidden, output_projection=self.output_proj, feed_previous=do_decode)

                outputs.append(bucket_outputs)
                losses.append(tf.nn.seq2seq.sequence_loss(outputs[-1], self.targets[:bucket[1]],
                                                          self.target_weights[:bucket[1]],
                                                          softmax_loss_function=
                                                          self.sampled_softmax_func))
        # Return Outputs, Losses
        return outputs, losses

    def train(self):
        """
        Build the training operation computation graph.
        """
        params = tf.trainable_variables()
        self.gradient_norms, self.updates = [], []
        opt = tf.train.AdagradOptimizer(self.learning_rate)

        for b in xrange(len(self.buckets)):
            gradients = tf.gradients(self.losses[b], params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(zip(clipped_gradients, params),
                                                    global_step=self.global_step))

    def get_batch(self, data, bucket_id):
        """
        Get a batch from the specified bucket, and prepare it for the training step.

        :param data: A tuple of size len(self.buckets) in which each element contains X, Y data for
                     given bucket.
        :param bucket_id: Bucket to get batch data for.
        :return: Tuple of (encoder_inputs, decoder_inputs, target_weights)
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data, pad them if needed, reverse
        # encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            encoder_pad = [reader.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([reader.GO_ID] + decoder_input +
                                  [reader.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx]
                                                  for batch_idx in xrange(self.batch_size)],
                                                 dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx]
                                                  for batch_idx in xrange(self.batch_size)],
                                                 dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == reader.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def step(self, sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
        """
        Run a step of the model, with the given inputs.

        :param sess: Tensorflow session to use.
        :param encoder_inputs: List of numpy int vectors to feed as input (source sentence)
        :param decoder_inputs: List of numpy int vectors to feed as decoder input (target sentence)
        :param target_weights: List of numpy float vectors to weight targets by (for padding)
        :param bucket_id: Which bucket of the model to use.
        :param forward_only: Whether to do the train (backward) pass, or not.
        :return: Tuple of (loss, intermediate embedding, outputs)
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket, %d != %d." %
                             (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket, %d != %d." %
                             (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket, %d != %d." %
                             (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id],  # Loss for this batch.
                           self.intermediate_embeddings[bucket_id]]  # Embeddings for this batch.
        else:
            output_feed = [self.losses[bucket_id],  # Loss for this batch.
                           self.intermediate_embeddings[bucket_id]]  # Embeddings for this batch.
            for l in xrange(decoder_size):  # Output logits
                output_feed.append(self.outputs[bucket_id][l])

        # Run, and get outputs
        outputs = sess.run(output_feed, feed_dict=input_feed)

        if not forward_only:
            return outputs[2], outputs[3], None  # Loss, embedding, no outputs.
        else:
            return outputs[0], outputs[1], outputs[2:]  # Loss, embedding, outputs.