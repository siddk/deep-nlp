"""
deep_mnist.py

Code for following Tensorflow Tutorial.
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Load MNIST Data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Start Tensorflow Session
sess = tf.InteractiveSession()

# Build Input/Output placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# # Single Layer Model
#
# # Set up Model Parameters
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
#
# # Initialize all variables in the session
# sess.run(tf.initialize_all_variables())
#
# # Actually implement the model
# y = tf.nn.softmax(tf.matmul(x, W) + b)
#
# # Set up loss function
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#
# # Start the training process
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
# # Train the model
# for i in range(1000):
#     batch = mnist.train.next_batch(50)
#     train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#
# # Set up evaluations
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Evaluate model
# print "Single Layer Accuracy:", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})


# Convolution Model
# Define variables
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Define special layers
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Build model
# Reshape Input
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolution + Max Pool (1)
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolution + Max Pool (2)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Dense Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32) # Placeholder lets us turn on during train, off during test
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train and Evaluate
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("Step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("Test accuracy: %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels,
                                                     keep_prob: 1.0}))