# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import gym

env = gym.make('Breakout-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())


class DQN():
    def __init__(self, FRAME_WIDTH=84, FRAME_HEIGHT=84, STATE_LENGTH=4, num_actions=1):
        self.frame_width = FRAME_WIDTH
        self.frame_height = FRAME_HEIGHT
        self.state_length = STATE_LENGTH
        self.num_actions = num_actions

    def preprocess(observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(
            resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
        return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))

    def network():
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu',
                                input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        s = tf.placeholder(
            tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
        q_values = model(s)

        return s, q_values, model


def main():
    dqn = DQN(num_actions=env.action_space.n)
