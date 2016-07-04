"""
train.py

Core file for training a Variational Autoencoder to generate MNIST handwritten digits. Takes an
input from the training set, maps it to latent space (z), then uses the latent space representation
to regenerate the input.
"""
from model.vae import VAE
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

# Set up Tensorflow Training Parameters, or Flags
FLAGS = tf.flags.FLAGS

# Set data parameters
tf.app.flags.DEFINE_string('log_dir', 'log/checkpoints/', "Directory to write checkpoints.")

# General Model Parameters
tf.app.flags.DEFINE_integer('x_dim', 784, "Dimensionality of input (an MNIST image is 784 pixels.")
tf.app.flags.DEFINE_integer('z_dim', 2, "Dimensionality of latent space.")
tf.app.flags.DEFINE_integer('batch_size', 100, "Mini-batch size for training.")
tf.app.flags.DEFINE_float('learning_rate', .001, "Learning rate for training.")
tf.app.flags.DEFINE_integer('epochs', 15, 'Number of epochs of training.')
tf.app.flags.DEFINE_integer('display_step', 2, 'Interval to print training progress.')

# Set Recognition Layer Parameters
tf.app.flags.DEFINE_integer('hidden_recog_1', 500, "Size of first recognition layer.")
tf.app.flags.DEFINE_integer('hidden_recog_2', 500, "Size of second recognition layer.")

# Set Generation Layer Parameters
tf.app.flags.DEFINE_integer('hidden_gen_1', 500, "Size of first generation layer.")
tf.app.flags.DEFINE_integer('hidden_gen_2', 500, "Size of second generation layer.")


def main(_):
    """
    Main training function, loads MNIST training data, then builds and trains the VAE. Also runs
    some plotting code in order to visually evaluate the quality of the reconstructions.
    """
    # Fix directories
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_samples = mnist.train.num_examples

    # Start training
    print "Starting Session"
    with tf.Session() as sess:
        # Instantiate Network
        vae = VAE()

        # Create a saver
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        # Run through the epochs
        for epoch in range(FLAGS.epochs):
            avg_cost = 0.
            total_batch = n_samples / FLAGS.batch_size

            # Loop over batches
            for i in range(total_batch):
                batch_x, _ = mnist.train.next_batch(FLAGS.batch_size)
                cost, _ = sess.run([vae.loss_val, vae.train_op], feed_dict={vae.X: batch_x})
                avg_cost += cost / (n_samples * FLAGS.batch_size)

            # Display step
            if epoch % FLAGS.display_step == 0:
                print "Epoch:", epoch, " " * 4, "Average Cost:", avg_cost
                checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, epoch)

        # Generate Reconstructed Pictures
        if FLAGS.z_dim > 2:
            x_sample = mnist.test.next_batch(FLAGS.batch_size)[0]
            x_reconstruct = vae.reconstruct(sess, x_sample)

            plt.figure(figsize=(8, 12))
            for i in range(5):
                plt.subplot(5, 2, 2 * i + 1)
                plt.imshow(x_sample[i+10].reshape(28, 28), vmin=0, vmax=1)
                plt.title("Test input")
                plt.colorbar()
                plt.subplot(5, 2, 2 * i + 2)
                plt.imshow(x_reconstruct[i+10].reshape(28, 28), vmin=0, vmax=1)
                plt.title("Reconstruction")
                plt.colorbar()
            plt.tight_layout()
            plt.show()
        else:
            nx = ny = 20
            x_values = np.linspace(-3, 3, nx)
            y_values = np.linspace(-3, 3, ny)
            canvas = np.empty((28 * ny, 28 * nx))
            for i, yi in enumerate(x_values):
                for j, xi in enumerate(y_values):
                    z_mu = np.tile(np.array([[xi, yi]]), (FLAGS.batch_size, 1))
                    x_mean = vae.generate(sess, z_mu)
                    canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

            plt.figure(figsize=(8, 10))
            Xi, Yi = np.meshgrid(x_values, y_values)
            plt.imshow(canvas, origin="upper")
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    tf.app.run()

