# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import gym
from skimage.color import rgb2gray
from skimage.transform import resize

env = gym.make('Breakout-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())


class DQN():
    def __init__(self, frame_width=84, frame_height=84, state_7=4, num_actions=1):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.state_length = state_length
        self.num_actions = num_actions

    def preprocess(observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(
            resize(rgb2gray(processed_observation), (self.frame_width, self.frame_height)) * 255)
        return np.reshape(processed_observation, (1, self.frame_width, self.frame_height))

    def network():
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu',
                                input_shape=(self.state_length, self.frame_width, self.frame_height)))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        s = tf.placeholder(
            tf.float32, [None, self.state_length, self.frame_width, self.frame_height])
        q_values = model(s)

        return s, q_values, model


def main():
    dqn = DQN(num_actions=env.action_space.n)
