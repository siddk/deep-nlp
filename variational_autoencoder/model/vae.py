"""
vae.py

Core script for defining the Variational Autoencoder model. Builds a Variational Autoencoder in
two parts:
  1) Recognition: Map an existing input example from the data set to latent space (z). (Normal)
  2) Generation: Go from a latent space vector (z) to a reconstructed/generated example. (Bernoulli)
"""
import numpy as np
import tensorflow as tf

# Get Tensorflow Parameters
FLAGS = tf.flags.FLAGS


class VAE:
    def __init__(self, activation=tf.nn.softplus):
        """
        Instantiate a Variational Autoencoder model, with the necessary hyperparameters.
        """
        self.activation_func = activation

        # Setup Placeholders
        self.X = tf.placeholder(tf.float32, [None, FLAGS.x_dim])  # Input is (bsz, 784)

        # Instantiate Weights
        self.instantiate_weights()

        # Use recognition network to determine mean and (log) variance of Gaussian Distribution in
        # latent space
        self.z_mean, self.z_log_sigma_sq = self.build_recognition_net()

        # Build the underlying latent space distribution, and sample from it. This is done very
        # simply, by drawing z = mu + sigma * noise
        self.noise = tf.random_normal([FLAGS.batch_size, FLAGS.z_dim], 0, 1, dtype=tf.float32)
        self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), self.noise))

        # Use the generator to determine the mean of Bernoulli Distribution of reconstructed input.
        self.x_reconstructed_mean = self.build_generator_net()

        # Evaluate loss/objective function
        self.loss_val = self.loss()

        # Build the training operation graph
        self.train_op = self.train()

    def instantiate_weights(self):
        """
        Create weight parameters (all trainable variables).
        """
        # Recognition Weights
        self.recog_h1 = tf.Variable(xavier_init(FLAGS.x_dim, FLAGS.hidden_recog_1))
        self.recog_b1 = tf.Variable(tf.zeros([FLAGS.hidden_recog_1], dtype=tf.float32))
        self.recog_h2 = tf.Variable(xavier_init(FLAGS.hidden_recog_1, FLAGS.hidden_recog_2))
        self.recog_b2 = tf.Variable(tf.zeros([FLAGS.hidden_recog_2], dtype=tf.float32))
        self.recog_out_mean = tf.Variable(xavier_init(FLAGS.hidden_recog_2, FLAGS.z_dim))
        self.recog_out_mean_bias = tf.Variable(tf.zeros([FLAGS.z_dim], dtype=tf.float32))
        self.recog_log_sigma = tf.Variable(xavier_init(FLAGS.hidden_recog_2, FLAGS.z_dim))
        self.recog_log_sigma_bias = tf.Variable(tf.zeros([FLAGS.z_dim], dtype=tf.float32))

        # Generation Weights
        self.gen_h1 = tf.Variable(xavier_init(FLAGS.z_dim, FLAGS.hidden_gen_1))
        self.gen_b1 = tf.Variable(tf.zeros([FLAGS.hidden_gen_1], dtype=tf.float32))
        self.gen_h2 = tf.Variable(xavier_init(FLAGS.hidden_gen_1, FLAGS.hidden_gen_2))
        self.gen_b2 = tf.Variable(tf.zeros([FLAGS.hidden_gen_2], dtype=tf.float32))
        self.gen_out_mean = tf.Variable(xavier_init(FLAGS.hidden_gen_2, FLAGS.x_dim))
        self.gen_out_mean_bias = tf.Variable(tf.zeros([FLAGS.x_dim], dtype=tf.float32))

    def build_recognition_net(self):
        """
        Build the recognition network (probabilistic encoder), which maps inputs to a normal
        distribution in the latent space (z_dim).

        :return Tuple of (z_mean, z_log_var), the mean and variance of this normal distribution.
        """
        layer_1 = self.activation_func(tf.add(tf.matmul(self.X, self.recog_h1), self.recog_b1))
        layer_2 = self.activation_func(tf.add(tf.matmul(layer_1, self.recog_h2), self.recog_b2))

        z_mean = tf.add(tf.matmul(layer_2, self.recog_out_mean), self.recog_out_mean_bias)
        z_log_sigma_sq = tf.add(tf.matmul(layer_2, self.recog_log_sigma), self.recog_log_sigma_bias)

        return z_mean, z_log_sigma_sq

    def build_generator_net(self):
        """
        Build the generator network (probabilistic decoder), which maps from the latent space to
        a Bernoulli distribution on the data space.
        """
        layer_1 = self.activation_func(tf.add(tf.matmul(self.z, self.gen_h1), self.gen_b1))
        layer_2 = self.activation_func(tf.add(tf.matmul(layer_1, self.gen_h2), self.gen_b2))

        x_reconstruction_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.gen_out_mean),
                                                     self.gen_out_mean_bias))
        return x_reconstruction_mean

    def loss(self):
        """
        Evaluate the loss function, which is the sum of two terms:
            1) The reconstruction loss (negative log probability of input under the reconstructed
               Bernoulli distribution).
            2) The latent loss, the KL Divergence between the distribution in latent space and some
               prior (in this case a standard normal).
        """
        # Reconstruction loss
        reconstruction_loss = -tf.reduce_sum(self.X * tf.log(1e-10 + self.x_reconstructed_mean) +
                                             (1 - self.X) * tf.log(1e-10 + 1 - self.x_reconstructed_mean),
                                             1)

        # Latent loss
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) -
                                           tf.exp(self.z_log_sigma_sq),
                                           1)

        # Return sum of respective losses
        return tf.reduce_mean(reconstruction_loss + latent_loss)

    def train(self):
        """
        Build training computation graph, using the loss value.
        """
        return tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.loss_val)

    def transform(self, sess, X):
        """
        Transform data by mapping it into the latent space (z).

        :param X: Input Feature Vector (from MNIST)
        :return Representation in Z dimensional space (maps to mean of Normal Distribution).
        """
        return sess.run(self.z_mean, feed_dict={self.X: X})

    def generate(self, sess, z_mu=None):
        """
        Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is generated. Otherwise,
        z_mu is drawn from prior in latent space.

        :return Generated input in data space (a handwritten digit).
        """
        if z_mu is None:
            z_mu = np.random.normal(size=FLAGS.z_dim)

        return sess.run(self.x_reconstructed_mean, feed_dict={self.z: z_mu})

    def reconstruct(self, sess, X):
        """
        Use the Variational Autoencoder to map input into latent space using the recognizer, then
        reconstruct input from latent space using the generator.

        :param X: Input to reconstruct (from MNIST).
        :return Reconstructed input (a handwritten digit).
        """
        return sess.run(self.x_reconstructed_mean, feed_dict={self.X: X})


def xavier_init(fan_in, fan_out, constant=1):
    """
    Xavier initialization of network weights
    - https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow

    :param fan_in: Input Dimension
    :param fan_out: Output Dimension
    :return Xavier initialized Tensor with shape (fan_in, fan_out)
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
