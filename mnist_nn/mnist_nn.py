"""
mnist_nn.py

Code for MNIST Tensorflow Tutorial. Builds a simple one layer network with a simple softmax.
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Load MNIST Data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Start Tensorflow Session
sess = tf.InteractiveSession()

# Build Input/Output placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])  # X is input, of shape (Batch, 784)
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # Y is output, of shape (Batch, 10)

# Single Layer Model

# Set up Model Parameters
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Initialize all variables in the session
sess.run(tf.initialize_all_variables())

# Actually implement the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Set up loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Start the training process
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Train the model
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Set up evaluations
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Evaluate model
print "Single Layer Accuracy:", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})