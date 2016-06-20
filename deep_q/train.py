"""
train.py

Core runner for Deep Q Network for learning to play Atari Pong.

Credit: https://github.com/DanielSlater/PyGamePlayer/blob/master/examples/deep_q_pong_player.py
"""
from model.q_pong import DeepQPongPlayer
import tensorflow as tf

# Set up Tensorflow Training Parameters, or "FLAGS"
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_actions', 3, "Number of available actions: still, up, down.")
tf.app.flags.DEFINE_integer('state_frames', 4, "Number of frames to store in state memory.")
tf.app.flags.DEFINE_integer('x_size', 80, "Width of the screen, in pixels.")
tf.app.flags.DEFINE_integer('y_size', 80, "Length of the screen, in pixels.")
tf.app.flags.DEFINE_float('future_reward_discount', 0.99, 'Decay rate of past observations.')
tf.app.flags.DEFINE_float('learning_rate', 1e-6, 'Learning rate for SGD.')
tf.app.flags.DEFINE_float('initial_random_action_prob', 1.0, "Start probability of random action.")
tf.app.flags.DEFINE_float('final_random_action_prob', .05, "Final probability of random action.")
tf.app.flags.DEFINE_float('store_scores_len', 200., "Size of score memory.")
tf.app.flags.DEFINE_integer('memory_size', 500000, "Size of observation memory.")
tf.app.flags.DEFINE_integer('observation_steps', 50000, "Time steps to observe before training.")
tf.app.flags.DEFINE_float('explore_steps', 500000., "Frames over which to anneal epsilon.")
tf.app.flags.DEFINE_integer('mini_batch_size', 100, 'Size of a mini-batch.')
tf.app.flags.DEFINE_integer('save_steps', 10000, "Interval between model serialization.")

if __name__ == "__main__":
    player = DeepQPongPlayer()
    player.start()