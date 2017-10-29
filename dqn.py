# coding:utf-8

import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras import backend as K
K.set_image_dim_ordering('th')
import argparse

BATCH_SIZE = 32
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
# Constant added to the squared gradient in the denominator of the RMSProp update
MIN_GRAD = 0.01


class Agent(object):
    def __init__(self, num_actions):
        self.frame_width = 84  # Resized frame width
        self.frame_height = 84
        self.state_length = 4

        self.init_replay_size = 20000
        self.target_update_interval = 10000
        self.save_interval = 300000

        self.num_actions = num_actions
        self.epsilon = 1.0
        self.epsilon_step = (
            1.0 - 0.1) / 1000000
        self.t = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(
            q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(
            q_network_weights)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(q_network_weights)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/Breakout-v0', self.sess.graph)

        if not os.path.exists('saved_networks/Breakout-v0'):
            os.makedirs('saved_networks/Breakout-v0')

        self.sess.run(tf.global_variables_initializer())

        # Load network
        self.load_training = False
        if self.load_training:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)

    def build_network(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=(self.state_length, self.frame_width, self.frame_height)))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        s = tf.placeholder(
            tf.float32, [None, self.state_length, self.frame_width, self.frame_height])
        q_values = model(s)

        return s, q_values, model

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(
            tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(
            LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grads_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grads_update

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(
            resize(rgb2gray(processed_observation), (self.frame_width, self.frame_height)) * 255)
        state = [processed_observation for _ in range(self.state_length)]
        return np.stack(state, axis=0)

    def get_action(self, state):
        if self.epsilon >= random.random() or self.t < self.init_replay_size:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(
                feed_dict={self.s: [np.float32(state / 255.0)]}))

        # Anneal epsilon linearly over time
        if self.epsilon > 0.1 and self.t >= self.init_replay_size:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state, action, reward, terminal, observation):
        next_state = np.append(state[1:, :, :], observation, axis=0)

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        reward = np.clip(reward, -1, 1)

        # Store transition in replay memory
        self.replay_memory.append(
            (state, action, reward, next_state, terminal))
        if len(self.replay_memory) > 400000:
            self.replay_memory.popleft()

        if self.t >= self.init_replay_size:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()

            # Update target network
            if self.t % self.target_update_interval == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % self.save_interval == 0:
                save_path = self.saver.save(
                    self.sess, 'saved_networks/Breakout-v0' + '/' + 'Breakout-v0', global_step=self.t)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(
            feed_dict={self.s: [np.float32(state / 255.0)]}))
        self.duration += 1

        if terminal:
            # Write summary
            if self.t >= self.init_replay_size:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                         self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL))]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < self.init_replay_size:
                mode = 'random'
            elif self.init_replay_size <= self.t < self.init_replay_size + 1000000:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_values.eval(
            feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)})
        y_batch = reward_batch + (1 - terminal_batch) * \
            0.99 * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: np.float32(np.array(state_batch) / 255.0),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar('Breakout-v0' + '/Total Reward/Episode',
                          episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(
            'Breakout-v0' + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(
            'Breakout-v0' + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(
            'Breakout-v0' + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward,
                        episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(
            tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(
            summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(
            'saved_networks/Breakout-v0')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_at_test(self, state):
        if random.random() <= 0.05:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(
                feed_dict={self.s: [np.float32(state / 255.0)]}))

        self.t += 1

        return action


def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(
        resize(rgb2gray(processed_observation), (84, 84)) * 255)
    return np.reshape(processed_observation, (1, 84, 84))


def main():
    parser = argparse.ArgumentParser(description="Perform DQN ATARI Training.",
                                     fromfile_prefix_chars="@", conflict_handler="resolve")

    parser.add_argument("--train", default=False, action='store_true',
                        help="Perform training; Else testing. [Default: %(default)s]")

    args = parser.parse_args()

    env = gym.make('Breakout-v0')
    agent = Agent(num_actions=env.action_space.n)

    if args.train:  # Train mode
        for _ in range(1200):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, 30)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_action(state)
                observation, reward, terminal, _ = env.step(action)
                # env.render()
                processed_observation = preprocess(
                    observation, last_observation)
                state = agent.run(state, action, reward,
                                  terminal, processed_observation)
    else:  # Testing
        for _ in range(20):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, 30)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_action_at_test(state)
                observation, _, terminal, _ = env.step(action)
                env.render()
                processed_observation = preprocess(
                    observation, last_observation)
                state = np.append(
                    state[1:, :, :], processed_observation, axis=0)
        # env.monitor.close()


if __name__ == '__main__':
    main()
