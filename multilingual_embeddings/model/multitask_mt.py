"""
multitask_mt.py

Core model definition file for constructing a many-to-many Machine Translation model.
"""
from preprocessor import reader
import numpy as np
import random
import tensorflow as tf


class MultiTaskMT:
    def __init__(self, source_ids, target_ids, max_vocab_size, hidden_size, num_layers, buckets,
                 learning_rate, learning_rate_decay_factor, max_gradient_norm, batch_size,
                 reverse=True, regularize=True, num_samples=512, forward_only=False):
        """
        Initialize a Sequence-to-Sequence Encoder-Decoder Multitask Model, for Machine Translation.

        :param source_ids: List of source language identifiers to build encoders for.
        :param target_ids: List of target language identifiers to build decoders for.
        :param max_vocab_size: Dictionary mapping language id to vocabulary size.
        :param hidden_size: Size of the internal LSTM layers.
        :param num_layers: Number of internal layers for each LSTM (encoder/decoder).
        :param buckets: List of bucket tuples denoting the source/target bucket splits.
        :param learning_rate: Learning rate for SGD.
        :param learning_rate_decay_factor: Decay factor for learning rate.
        :param max_gradient_norm: Norm to clip gradients to (to prevent exploding gradients).
        :param batch_size: Batch size, for training.
        :param reverse: Boolean swap source and targets, and build enc/dec in opposite direction.
        :param regularize: Boolean add same-same language paths (i.e. english-english).
        :param num_samples: Number of samples to take for sampled softmax.
        """
        self.source_ids, self.target_ids = source_ids, target_ids
        self.max_vocab_size, self.paths = max_vocab_size, {}
        self.hidden, self.num_layers = hidden_size, num_layers
        self.buckets = buckets
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate *
                                                                learning_rate_decay_factor)
        self.max_gradient_norm = max_gradient_norm
        self.batch_size = batch_size
        self.global_step = tf.Variable(0, trainable=False)
        self.reverse, self.regularize = reverse, regularize

        # Setup sampled softmax parameters
        self.sampled_softmax = self.build_sampled_softmax(num_samples)

        # Set up placeholders -> Encoder/Decoder takes a list of inputs, not a tensor
        self.encoder_inputs = {lang_id: [] for lang_id in self.source_ids}
        self.decoder_inputs = {lang_id: [] for lang_id in self.target_ids}
        self.target_weights = {lang_id: [] for lang_id in self.target_ids}
        self.targets = {lang_id: [] for lang_id in self.target_ids}
        if reverse or regularize:
            self.encoder_inputs.update({lang_id: [] for lang_id in self.target_ids})
            self.decoder_inputs.update({lang_id: [] for lang_id in self.source_ids})
            self.target_weights.update({lang_id: [] for lang_id in self.source_ids})
            self.targets.update({lang_id: [] for lang_id in self.source_ids})

        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            for s in self.encoder_inputs:
                self.encoder_inputs[s].append(tf.placeholder(tf.int32, shape=[None],
                                                             name="{0}_encoder{1}".format(s, i)))
        for i in xrange(buckets[-1][1] + 1):
            for t in self.decoder_inputs:
                self.decoder_inputs[t].append(tf.placeholder(tf.int32, shape=[None],
                                                             name="{0}_decoder{1}".format(t, i)))
                self.target_weights[t].append(tf.placeholder(tf.float32, shape=[None],
                                                             name="{0}_weight{1}".format(t, i)))

        # Targets are decoder inputs shifted by one.
        for t in self.targets:
            self.targets[t] = [self.decoder_inputs[t][i + 1] for i in
                               xrange(len(self.decoder_inputs[t]) - 1)]

        # Instantiate Weights
        self.instantiate_weights()

        # Build inference pipeline for both training and evaluation
        if forward_only:
            pass
        else:
            self.graphs, self.embeddings = self.inference(False)

        # Set up training graph
        self.train()

        # Set up saver
        self.saver = tf.train.Saver(tf.all_variables())

    def build_sampled_softmax(self, num_samples):
        """
        Setup the output projection parameters for the calculation of the sampled softmax loss
        function.

        :param num_samples: Number of samples for sampled softmax
        :return Dictionary mapping target language id to tuple of (weight, bias), softmax function
        """
        targets = set(self.target_ids)
        if self.reverse:
            targets.update(set(self.source_ids))
        print "Building output projections for languages:", targets

        sampled_softmax_dict = {}
        for t in targets:
            with tf.variable_scope('decoder_softmax_scope'):
                target_vsz = self.max_vocab_size[t]
                w = tf.get_variable("{0}_proj_w".format(t), [self.hidden, target_vsz])
                w_t = tf.transpose(w)
                b = tf.get_variable("{0}_proj_b".format(t), [target_vsz])

                def sampled_loss(inputs, labels):
                    labels = tf.reshape(labels, [-1, 1])
                    return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, target_vsz)

                sampled_softmax_dict[t] = (w, b), sampled_loss
        return sampled_softmax_dict

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
        :return: Nested dictionary mapping paths to tuple of RNN Outputs, and RNN Losses.
        """
        # Build all the input/output paths
        logit_paths = {}
        for s in self.source_ids:
            logit_paths[s] = {}
            for t in self.target_ids:
                logit_paths[s][t] = {}
        if self.reverse:
            for t in self.target_ids:
                if not t in logit_paths:
                    logit_paths[t] = {}
                for s in self.source_ids:
                    logit_paths[t][s] = {}
        if self.regularize:
            for s in logit_paths:
                logit_paths[s][s] = {}

        # Compile the source encoders
        print "Compiling Inference Graphs!"
        embeddings = {k: [] for k in logit_paths}
        for source_id in logit_paths:
            with tf.variable_scope('{0}_encoder_scope'.format(source_id)):
                for j, bucket in enumerate(self.buckets):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 else None):
                        encoder_cell = tf.nn.rnn_cell.EmbeddingWrapper(self.cell,
                                                                       embedding_classes=
                                                                       self.max_vocab_size[source_id],
                                                                       embedding_size=self.hidden)

                        # Run the inputs through the encoder cell
                        _, encoder_state = tf.nn.rnn(encoder_cell,
                                                     self.encoder_inputs[source_id][:bucket[0]],
                                                     dtype=tf.float32)

                        # Transform the encoder state through a ReLU Hidden Layer
                        inter_state = tf.matmul(encoder_state, self.inter_weight) + self.inter_bias
                        inter_relu = tf.nn.relu(inter_state, "inter_relu")

                        # Linear transform, get and append intermediate embedding
                        embedding = tf.matmul(inter_relu, self.embed_weight) + self.embed_bias
                        embeddings[source_id].append(embedding)

        targets = set(reduce(lambda x, y: x + y, [logit_paths[k].keys() for k in logit_paths]))
        for target_id in targets:
            # Get all possible source combinations for the given target.
            source_set = [k for k in logit_paths if target_id in logit_paths[k]]
            # For each source id in the source set for the given target language.
            for i, source_id in enumerate(source_set):
                # Reuse the variable scope for the given target language
                with tf.variable_scope('{0}_decoder_scope'.format(target_id), reuse=True if i > 0
                                                                                         else None):
                    # Initialize the path outputs (actual outputs, losses)
                    logit_paths[source_id][target_id]['outputs'] = []
                    logit_paths[source_id][target_id]['losses'] = []

                    # For each bucket
                    for j, bucket in enumerate(self.buckets):
                        # Reuse the variables for the given source-target pair
                        with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0
                                                                                   else None):
                            # Get decoder outputs
                            bucket_outputs, _ = tf.nn.seq2seq.embedding_rnn_decoder(
                                self.decoder_inputs[target_id][:bucket[1]],
                                embeddings[source_id][j], self.cell, self.max_vocab_size[target_id],
                                self.hidden, output_projection=self.sampled_softmax[target_id][0],
                                feed_previous=do_decode)

                            logit_paths[source_id][target_id]['outputs'].append(bucket_outputs)
                            logit_paths[source_id][target_id]['losses'].append(
                                tf.nn.seq2seq.sequence_loss(bucket_outputs,
                                                            self.targets[target_id][:bucket[1]],
                                                            self.target_weights[target_id]
                                                                               [:bucket[1]],
                                                            softmax_loss_function=
                                                            self.sampled_softmax[target_id][1])
                            )
                    print "Compiled entire %s-%s inference graph!" % (source_id, target_id)

        return logit_paths, embeddings

    def train(self):
        """
        Build the training operation computation graph.
        """
        params = tf.trainable_variables()
        opt = tf.train.AdagradOptimizer(self.learning_rate)

        for source_id in self.graphs:
            for target_id in self.graphs[source_id]:
                self.graphs[source_id][target_id]['gradient norms'] = []
                self.graphs[source_id][target_id]['updates'] = []
                for b in xrange(len(self.buckets)):
                    gradients = tf.gradients(self.graphs[source_id][target_id]['losses'][b], params)
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                     self.max_gradient_norm)
                    self.graphs[source_id][target_id]['gradient norms'].append(norm)
                    self.graphs[source_id][target_id]['updates'].append(
                        opt.apply_gradients(zip(clipped_gradients, params),
                                            global_step=self.global_step))
                print "Initialized Training Operation for %s-%s Graph!" % (source_id, target_id)

    def get_batch(self, bucket_data, bucket_id):
        """
        Get a batch from the specified bucket, and prepare it for the training step.

        :param bucket_data: A list in which each element contains X, Y data for the given bucket.
        :param bucket_id: Bucket to get batch data for.
        :return: Tuple of (encoder_inputs, decoder_inputs, target_weights)
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data, pad them if needed, reverse
        # encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(bucket_data)

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

    def step(self, sess, encoder_inputs, decoder_inputs, target_weights, pair, bucket_id, forward):
        """
        Run a step of the model, with the given inputs.

        :param sess: Tensorflow session to use.
        :param encoder_inputs: List of numpy int vectors to feed as input (source sentence)
        :param decoder_inputs: List of numpy int vectors to feed as decoder input (target sentence)
        :param target_weights: List of numpy float vectors to weight targets by (for padding)
        :param pair: String of form "en-de" representing the source and target of the given step.
        :param bucket_id: Which bucket of the model to use.
        :param forward: Whether to do the train (backward) pass, or not.
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

        # Figure out the source and target languages.
        s, t = pair.split('-')

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[s][l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[t][l].name] = decoder_inputs[l]
            input_feed[self.target_weights[t][l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[t][decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward:
            output_feed = [self.graphs[s][t]['updates'][bucket_id],         # Update Op of SGD.
                           self.graphs[s][t]['gradient norms'][bucket_id],  # Gradient norm.
                           self.graphs[s][t]['losses'][bucket_id],  # Loss for this batch.
                           self.embeddings[s][bucket_id]]           # Embeddings for this batch.
        else:
            output_feed = [self.graphs[s][t]['losses'][bucket_id],  # Loss for this batch.
                           self.embeddings[s][bucket_id]]           # Embeddings for this batch.
            for l in xrange(decoder_size):  # Output logits
                output_feed.append(self.graphs[s][t]['outputs'][bucket_id][l])

        # Run, and get outputs
        outputs = sess.run(output_feed, feed_dict=input_feed)

        if not forward:
            return outputs[2], outputs[3], None         # Loss, embedding, no outputs.
        else:
            return outputs[0], outputs[1], outputs[2:]  # Loss, embedding, outputs.

    def eval(self, sess, source_data, target_data, source_id, target_id, bucket_id):
        """
        Evaluate the quality of the sentence embeddings by computing the average cosine similarity
        for each bucket on the evaluation data.

        :param sess: Tensorflow Session
        :param source_data: Matrix of source data (num_examples x bucket_size)
        :param target_data: Matrix of target data (num_examples x bucket_size)
        :param source_id: Source language identifier
        :param target_id: Target language identifier
        :param bucket_id: Current bucket id
        """
        source_size, target_size = self.buckets[bucket_id]
        source_inputs, target_inputs = [], []

        for i in xrange(len(source_data)):
            source_input, target_input = source_data[i], target_data[i]

            # Encoder inputs are padded and then reversed.
            source_pad = [reader.PAD_ID] * (source_size - len(source_input))
            source_inputs.append(list(reversed(source_input + source_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            target_pad = [reader.PAD_ID] * (target_size - len(target_input))
            target_inputs.append(list(reversed(target_input + target_pad)))

        # Now we create batch-major vectors from the data selected above.
        batch_source_inputs, batch_target_inputs = [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(source_size):
            batch_source_inputs.append(np.array([source_inputs[batch_idx][length_idx]
                                                for batch_idx in xrange(200)],
                                                dtype=np.int32))

        for length_idx in xrange(target_size):
            batch_target_inputs.append(np.array([target_inputs[batch_idx][length_idx]
                                                for batch_idx in xrange(200)],
                                                dtype=np.int32))

        # Input feed: Source inputs, Target inputs
        input_feed = {}
        for l in xrange(source_size):
            input_feed[self.encoder_inputs[source_id][l].name] = batch_source_inputs[l]
        for l in xrange(target_size):
            input_feed[self.encoder_inputs[target_id][l].name] = batch_target_inputs[l]


        # Output feed: depends on whether we do a backward step or not.
        output_feed = [self.embeddings[source_id][bucket_id],
                       self.embeddings[target_id][bucket_id]]

        # Run, and get outputs
        outputs = sess.run(output_feed, feed_dict=input_feed)
        src_embeddings, trg_embeddings = outputs

        return src_embeddings, trg_embeddings

