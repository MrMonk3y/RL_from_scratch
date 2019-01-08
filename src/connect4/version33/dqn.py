#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 19:55:38 2018

@author: saschajecklin
"""

import sys
sys.path.append("..")
import random
import numpy as np
from collections import deque
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Conv2D
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation

GAMMA = 0.95

MEMORY_SIZE = 1000000
BATCH_SIZE = 512

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.998

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.observation_space =  observation_space
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        in_x = x = Input((self.observation_space[0], self.observation_space[1], 2))  # stack of own(6x7) and enemy(6x7) field

        x = Conv2D(128, 3, padding="same", kernel_regularizer=l2(1e-4),
                   data_format="channels_last")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        for _ in range(2):
            x = self._build_residual_block(x)

        x = Conv2D(filters=2, kernel_size=1, kernel_regularizer=l2(1e-4),
                   data_format="channels_last")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        policy_out = Dense(action_space, kernel_regularizer=l2(1e-4), activation="softmax", name="policy_out")(x)

        self.model = Model(in_x, policy_out, name="connect4_model")

        self.optimizer = RMSprop(lr=0.00025, rho=0.9, epsilon=1e-6, decay=0.0) #SGD(lr=1e-2, momentum=0.9)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')

    def _build_residual_block(self, x):
        in_x = x
        x = Conv2D(filters=128, kernel_size=3, padding="same",
                   kernel_regularizer=l2(1e-4), data_format="channels_last")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=128, kernel_size=3, padding="same",
                   kernel_regularizer=l2(1e-4), data_format="channels_last")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([in_x, x])
        x = Activation("relu")(x)
        return x

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # mirror state, next_state and action to produce twice as much training data
        self.memory.append((np.flip(state, 1), (self.action_space-1)-action, reward, np.flip(next_state, 1), done))

    def pop(self):
        for i in range(2): # pop 2 becauses mirrored entries in remeber()
            self.memory.pop()

    def act(self, state, env): # state doesnt have to be the state in env. could be inverted
        if np.random.rand() < self.exploration_rate:
            return env.sample()
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state)
        mask = (np.expand_dims(env.validMoves(),0) == 0)
        q_values[mask] = float('-inf') # guard for valid moves
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        state_batch = np.zeros((BATCH_SIZE, self.observation_space[0], self.observation_space[1], 2))
        q_values_batch = np.zeros((BATCH_SIZE, self.action_space))
        idx = 0
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                state_next = np.expand_dims(state_next, axis=0)
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            state = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            state_batch[idx, ...] = state
            q_values_batch[idx, ...] = q_values
            idx = idx + 1

        self.model.fit(state_batch, q_values_batch, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def save(self, path='weights.h5'):
        self.model.save_weights(filepath=path)

    def load(self, path='weights.h5'):
        self.model.load_weights(filepath=path)
