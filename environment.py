# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import tensorflow as tf
import gym

env = gym.make('Breakout-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
