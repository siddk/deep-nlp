"""
q_pong.py

Class definition file for the Deep Q Pong Network.

Credit: https://github.com/DanielSlater/PyGamePlayer/blob/master/examples/deep_q_pong_player.py
"""
from collections import deque
from PyGamePlayer.examples.pong_player import PongPlayer
from pygame.constants import K_DOWN, K_UP
import cv2
import numpy as np
import os
import random
import tensorflow as tf

# Get current parameters
FLAGS = tf.app.flags.FLAGS

# Set up index variables
OBS_LAST_STATE, OBS_ACTION, OBS_REWARD, OBS_CURRENT_STATE, OBS_TERMINAL = range(5)


class DeepQPongPlayer(PongPlayer):
    def __init__(self, checkpoint_path="deep_q_networks", playback_mode=False, verbose=False):
        """
        Initialize a DeepQ Model for learning Pong, with the necessary hyperparameters.

        :param checkpoint_path: Path to save trained models.
        :param playback_mode: Boolean if running in playback mode.
        :param verbose: Boolean for verbose logging.
        """
        super(DeepQPongPlayer, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.playback_mode, self.verbose = playback_mode, verbose
        self.state_frames, self.num_actions = FLAGS.state_frames, FLAGS.num_actions
        self.x_len, self.y_len = FLAGS.x_size, FLAGS.y_size
        self.future_reward_discount = FLAGS.future_reward_discount
        self.initial_random_action_prob = FLAGS.initial_random_action_prob
        self.mini_batch_size = FLAGS.mini_batch_size
        self.learning_rate = FLAGS.learning_rate
        self.session = tf.Session()

        # Initialize Placeholders
        self.X = tf.placeholder(tf.float32, shape=[None, self.x_len, self.y_len, self.state_frames])
        self.action = tf.placeholder(tf.float32, shape=[None, self.num_actions])
        self.target = tf.placeholder(tf.float32, shape=[None])

        # Instantiate all trainable weights
        self.build_weights()

        # Build inference pipeline (computation graph)
        self.logits = self.inference()

        # Build loss pipeline
        self.loss_val = self.loss()

        # Set up the training operation
        self.train_op = self.train()

        # Instantiate model memory parameters
        self.observations = deque()
        self.last_scores = deque()

        # Set the first action to do nothing
        self.last_action = np.zeros(self.num_actions)
        self.last_action[1] = 1

        self.last_state = None
        self.probability_of_random_action = self.initial_random_action_prob
        self.time = 0

        # Start the training process
        self.session.run(tf.initialize_all_variables())

        # Build out the checkpoint pipeline
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)

        if checkpoint and checkpoint.model_checkpoint_path:
            self._saver.restore(self._session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)
        elif self.playback_mode:
            raise Exception("Could not load checkpoints for playback")

    def build_weights(self):
        """
        Instantiate all the weight tensors for each step of the learning process.
        """
        self.conv1_weight = init_weight([8, 8, self.state_frames, 32], "Conv1_Weight")
        self.conv1_bias = init_bias(32, .01, "Conv1_Bias")

        self.conv2_weight = init_weight([4, 4, 32, 64], "Conv2_Weight")
        self.conv2_bias = init_bias(64, .01, "Conv2_Bias")

        self.conv3_weight = init_weight([3, 3, 64, 64], "Conv3_Weight")
        self.conv3_bias = init_bias(64, .01, "Conv3_Bias")

        self.ff1_weight = init_weight([256, 256], "FF1_Weight")
        self.ff1_bias = init_bias(256, .01, "FF1_Bias")

        self.ff2_weight = init_weight([256, self.num_actions], "FF2_Weight")
        self.ff2_bias = init_bias(self.num_actions, .01, "FF2_Bias")

    def inference(self):
        """
        Build computation graph, covering all the layers from the Input, to the final logits.
        """
        conv1 = tf.nn.relu(
            tf.nn.conv2d(self.X, self.conv1_weight, strides=[1, 4, 4, 1], padding="SAME")
            + self.conv1_bias
        )
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        conv2 = tf.nn.relu(
            tf.nn.conv2d(pool1, self.conv2_weight, strides=[1, 2, 2, 1], padding="SAME")
            + self.conv2_bias
        )
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        conv3 = tf.nn.relu(
            tf.nn.conv2d(pool2, self.conv3_weight, strides=[1, 1, 1, 1], padding="SAME")
            + self.conv3_bias
        )
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        flatten = tf.reshape(pool3, [-1, 256])
        hidden = tf.nn.relu(tf.matmul(flatten, self.ff1_weight) + self.ff1_bias)
        logits = tf.matmul(hidden, self.ff2_weight) + self.ff2_bias

        return logits

    def loss(self):
        """
        Build the computation graph for computing the model cost function.
        """
        readout_action = tf.reduce_sum(tf.mul(self.logits, self.action), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(self.target - readout_action))

        return cost

    def train(self):
        """
        Build the training operation graph.
        """
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)

    def get_keys_pressed(self, screen_array, reward, terminal):
        # Scale down game image
        screen_resized_grayscaled = cv2.cvtColor(cv2.resize(screen_array, (self.x_len, self.y_len)),
                                                 cv2.COLOR_BGR2GRAY)

        # Set the grayscale to have values in the 0.0 to 1.0 range
        _, screen_resized_binary = cv2.threshold(screen_resized_grayscaled, 1, 255,
                                                 cv2.THRESH_BINARY)

        # Update score (reward) memory
        if reward != 0.0:
            self.last_scores.append(reward)
            if len(self.last_scores) > FLAGS.store_scores_len:
                self.last_scores.popleft()

        # First frame must be handled differently
        if self.last_state is None:
            # The last_state will contain the image data from the last self.STATE_FRAMES frames
            self.last_state = np.stack(
                tuple(screen_resized_binary for _ in range(self.state_frames)),
                axis=2
            )
            return key_presses_from_action(self.last_action)

        screen_resized_binary = np.reshape(screen_resized_binary, (self.x_len, self.y_len, 1))
        current_state = np.append(self.last_state[:, :, 1:], screen_resized_binary, axis=2)

        if not self.playback_mode:
            # Store the transition in previous_observations
            self.observations.append((self.last_state,
                                      self.last_action,
                                      reward,
                                      current_state,
                                      terminal))

            if len(self.observations) > FLAGS.memory_size:
                self.observations.popleft()

            # Only train if done observing
            if len(self.observations) > FLAGS.observation_steps:
                self.run_training()
                self.time += 1

        # Update the old values
        self.last_state = current_state

        self.last_action = self.choose_next_action()

        if not self.playback_mode:
            # Gradually reduce the probability of a random action.
            if self.probability_of_random_action > FLAGS.final_random_action_prob \
                    and len(self.observations) > FLAGS.observation_steps:
                self.probability_of_random_action -= \
                    (FLAGS.initial_random_action_prob - FLAGS.final_random_action_prob) / \
                    FLAGS.explore_steps

            print("Time: %s random_action_prob: %s reward %s scores differential %s" %
                  (self.time, self.probability_of_random_action, reward,
                   sum(self.last_scores) / FLAGS.store_scores_len))

        return key_presses_from_action(self.last_action)

    def choose_next_action(self):
        """
        Choose either a random action, or run through the computation pipeline to decide on the
        next action.
        """
        new_action = np.zeros([self.num_actions])

        if (not self.playback_mode) and (random.random() <= self.probability_of_random_action):
            # Choose an action randomly
            action_index = random.randrange(self.num_actions)
        else:
            # Choose an action given our last state
            readout_t = self.session.run(self.logits, feed_dict={self.X: [self.last_state]})[0]
            if self.verbose:
                print("Action Q-Values are %s" % readout_t)
            action_index = np.argmax(readout_t)

        new_action[action_index] = 1
        return new_action

    def run_training(self):
        """
        Run a training iteration.
        """
        # Sample a mini_batch to train on
        mini_batch = random.sample(self.observations, self.mini_batch_size)

        # Get the batch variables
        previous_states = [d[OBS_LAST_STATE] for d in mini_batch]
        actions = [d[OBS_ACTION] for d in mini_batch]
        rewards = [d[OBS_REWARD] for d in mini_batch]
        current_states = [d[OBS_CURRENT_STATE] for d in mini_batch]
        agents_expected_reward = []

        # This gives us the agents expected reward for each action we might
        agents_reward_per_action = self.session.run(self.logits, feed_dict={self.X: current_states})
        for i in range(len(mini_batch)):
            if mini_batch[i][OBS_TERMINAL]:
                # This was a terminal frame so there is no future reward
                agents_expected_reward.append(rewards[i])
            else:
                agents_expected_reward.append(
                    rewards[i] + self.future_reward_discount * np.max(agents_reward_per_action[i]))

        # Learn that these actions in these states lead to this reward
        self.session.run(self.train_op, feed_dict={
            self.X: previous_states,
            self.action: actions,
            self.target: agents_expected_reward}
        )

        # Save checkpoints for later
        if self.time % FLAGS.save_steps == 0:
            self.saver.save(self.session, self.checkpoint_path + '/network', global_step=self.time)


def init_weight(shape, name):
    """
    Initialize a Tensor corresponding to a weight matrix with the given shape and name.

    :param shape: Shape of the weight tensor.
    :param name: Name of the weight tensor in the computation graph.
    :return: Tensor object with given shape and name, initialized from a standard normal.
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)


def init_bias(shape, value, name):
    """
    Initialize a Tensor corresponding to a bias vector with the given shape and name.

    :param shape: Shape of the bias vector (as an int, not a list).
    :param value: Value to initialize bias to.
    :param name: Name of the bias vector in the computation graph.
    :return: Tensor (Vector) object with given shape and name, initialized with given bias.
    """
    return tf.Variable(tf.constant(value, shape=[shape]), name=name)


def key_presses_from_action(action_set):
    """
    Utility function to turn action vector into actual Game actions.

    :param action_set: Set of actions (one-hot).
    :return: Actual in-game action to perform.
    """
    if action_set[0] == 1:
        return [K_DOWN]
    elif action_set[1] == 1:
        return []
    elif action_set[2] == 1:
        return [K_UP]
    raise Exception("Unexpected action")