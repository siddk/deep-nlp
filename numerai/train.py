"""
train.py
"""
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.cluster import KMeans
import csv
import numpy as np

DATA_PATH = "data/numerai_training_data.csv"
TEST_PATH = "data/numerai_tournament_data.csv"
NUM_CLUSTERS = 10


def load_data(path):
    """
    Read training data from path, preprocess, return labels and features as two separate tensors.

    :return: Tuple of feature tensor, label tensor.
    """
    x, y = [], []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            x.append(row[:-1])
            y.append(row[-1])
    return np.array(x[1:], dtype='float32'), np.array(y[1:], dtype='int32')


def partition_data(x):
    """
    Use K-means to partition the data into

    :param x: Feature tensor (shape [96320, 21])

    :return: K-means Model
    """
    k_means = KMeans(init='k-means++', n_clusters=NUM_CLUSTERS, n_init=5)
    k_means.fit(x)
    return k_means


def simple_nn(x, y, hidden=100, dropout=0.1, num_layers=4):
    """
    Build and train simple NN on the data set.

    :param x: Feature tensor (shape [96320, 21])
    :param y: Label tensor (shape [96320])

    :return: Trained Simple NN model
    """
    nn = Sequential()
    nn.add(Dense(hidden, input_dim=x.shape[1], activation='tanh'))
    nn.add(Dropout(dropout))
    for _ in range(num_layers - 1):
        nn.add(Dense(hidden, activation='relu'))
    nn.add(Dense(2, activation='softmax'))

    nn.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    nn.fit(x, to_categorical(y), validation_split=.01, shuffle=True)

    return nn


if __name__ == "__main__":
    # Read data, labels
    data, labels = load_data(DATA_PATH)

    # Partition the data into NUM_CLUSTERS clusters
    kmeans = partition_data(data)
    cluster_x = kmeans.predict(data)

    # Train NUM_CLUSTERS Different Neural Networks
    networks = []
    for i in range(NUM_CLUSTERS):
        raw_input("Continue: ")
        data_x = data[cluster_x == i, :]
        labels_y = labels[cluster_x == i]
        print data_x.shape, labels_y.shape
        networks.append(simple_nn(data_x, labels_y))

    # Evaluate and write Test Data
    test_data, test_ids = [], []
    with open(TEST_PATH, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            test_data.append(row[1:])
            test_ids.append(row[0])
    test_x, test_id = np.array(test_data[1:], dtype='float32'), np.array(test_ids[1:], dtype='int32')
    test_clusters = kmeans.predict(test_x)

    test_dict = {}
    for i in range(NUM_CLUSTERS):
        ttx = test_x[test_clusters == i, :]
        ttid = test_id[test_clusters == i]
        test_dict[i] = (networks[i].predict(ttx), ttid)

    with open('data/output.csv', 'w') as f:
        for k in test_dict:
            preds, tid = test_dict[k]
            for index in range(preds.shape[0]):
                f.write(str(tid[index]) + "," + str(preds[index][1]) + "\n")


