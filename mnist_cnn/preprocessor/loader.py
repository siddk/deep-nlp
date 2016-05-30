"""
loader.py

Core file for downloading, extracting, and formatting data for training/testing the MNIST
CNN Model.

Credit: Tensorflow Tutorial
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py
"""
from __future__ import division
import gzip
import numpy
import os
import tensorflow as tf
from six.moves import urllib

FLAGS = tf.app.flags.FLAGS


def load_data():
    """
    Download, load, and transform data into Numpy Arrays for training.

    :return: Tuple of train data, train labels, val data, val labels, and test data, test labels.
    """

    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, FLAGS.train_size + FLAGS.validation_size)
    train_labels = extract_labels(train_labels_filename, FLAGS.train_size + FLAGS.validation_size)
    test_data = extract_data(test_data_filename, FLAGS.test_size)
    test_labels = extract_labels(test_labels_filename, FLAGS.test_size)

    validation_data = train_data[:FLAGS.validation_size, ...]
    validation_labels = train_labels[:FLAGS.validation_size]
    train_data = train_data[FLAGS.validation_size:, ...]
    train_labels = train_labels[FLAGS.validation_size:]

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def maybe_download(filename):
    """
    Download the MNIST data from Yann LeCun's website, unless it already exists.

    :param filename: Name of the file to download.
    :return: Path to downloaded data
    """
    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    filepath = os.path.join(FLAGS.data_dir, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(FLAGS.source_url + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.Size()
        print('Successfully downloaded', filename, size, 'bytes.')

    return filepath


def extract_data(filename, num_images):
    """
    Extract the images into a 4D tensor of [image index, y, x, channels]. Values get rescaled
    from [0, 255] (RGB) to the range [-0.5, 0.5].

    :param filename: Path to data to extract and transform.
    :param num_images: Number of examples to pull from data set.
    :return: Numpy array representing data.
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(FLAGS.image_size * FLAGS.image_size * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (FLAGS.pixel_depth / 2.0)) / FLAGS.pixel_depth
        data = data.reshape(num_images, FLAGS.image_size, FLAGS.image_size, 1)
        return data


def extract_labels(filename, num_images):
    """
    Extract the data labels into a vector of int64 label IDs.

    :param filename: Path to labels to extract.
    :param num_images: Number of labels to pull from label set.
    :return: Numpy array representing labels.
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
    return labels