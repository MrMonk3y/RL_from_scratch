#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 19:55:38 2018

@author: saschajecklin
"""

import sys
import os
sys.path.append("..")
#from scores.score_logger import ScoreLogger
from connect4game import Connect4
import random
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Conv2D
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.backend.tensorflow_backend import set_session
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
import shutil

GAMMA = 0.95

MEMORY_SIZE = 1000000
BATCH_SIZE = 2048

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.998

NUMBER_OF_EVAL_GAMES = 10
SAVE_EVERY_K_GAMES = 12
GAMES_RECORDED_PER_EVAL = 10
DEMO_MODE = 0

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.observation_space =  observation_space
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        i = 0 #remove old Evaluations
        while os.path.exists("Evaluation%s" % i):
            shutil.rmtree("Evaluation%s" % i)
            i += 1

        in_x = x = Input((self.observation_space[0], self.observation_space[1], 2))  # stack of own(6x7) and enemy(6x7) field

        x = Conv2D(128, 3, padding="same", kernel_regularizer=l2(1e-4),
                   data_format="channels_last")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        for _ in range(2):
            x = self._build_residual_block(x)

        x = Conv2D(filters=2, kernel_size=1, padding="same", kernel_regularizer=l2(1e-4),
                   data_format="channels_last")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(256, kernel_regularizer=l2(1e-4), activation="relu")(x)
        policy_out = Dense(action_space, kernel_regularizer=l2(1e-4), activation="softmax", name="policy_out")(x)

        self.model = Model(in_x, policy_out, name="connect4_model")

        self.optimizer = RMSprop(lr=0.25, rho=0.95, epsilon=1e-2, decay=0.0) #SGD(lr=1e-2, momentum=0.9
#       'deepmind_rmsprop_learning_rate': 0.00025,
#       'deepmind_rmsprop_rho': .95,
#       'deepmind_rmsprop_epsilon': 0.01,)
#        self.optimizer = SGD(lr=1e-2, momentum=0.9)
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

def connect4dqn():
    env = Connect4()
    #score_logger = ScoreLogger('Connect4')
#    player1won = 0
#    player2won = 0
    observation_space = env.reset().shape
    action_space = env.validMoves().size
    # Assign GPU to DGX
    config = tf.ConfigProto(
        device_count = {'GPU': 2}
    )
    sess = tf.Session(config=config)
    set_session(sess)

    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    state = env.reset() #moved one loop up. otherwise player two wont be able to start if player one wins
    while True:
        run += 1
        #if run % SAVE_EVERY_K_GAMES == 0 :
        #    print('Saving weights and starting evaluation...')
        #    dqn_solver.save()
        #    score, ties = evaluate_dqn(env, dqn_solver, NUMBER_OF_EVAL_GAMES)
        #    score_logger.add_score(score + ties, run) #logging ties as success
        step = 0


        while True:
            step += 1
            player = env.getNextPlayer()

            if player == 1:
                action_player1 = dqn_solver.act(state, env)
                state_next, reward_player1, terminal, info = env.makeMove(player, action_player1, DEMO_MODE)
                state_copy = np.copy(state)
                state_next_copy = np.copy(state_next)
                if terminal:
                    dqn_solver.pop() # if player 1 wins, pop player 2's last move from and give it a negative reward
                    dqn_solver.remember(normalized_state, action_player2, reward_player1*-1, normalized_state_next, terminal)
                dqn_solver.remember(state, action_player1, reward_player1, state_next, terminal)
                state = state_next
            else:
                normalized_state = np.roll(state, 1, axis = -1)
                action_player2 = dqn_solver.act(normalized_state, env)
                state_next, reward_player2, terminal, info = env.makeMove(player, action_player2, DEMO_MODE)
                normalized_state_next = np.roll(state_next, 1, axis = -1)
                if terminal:
                    dqn_solver.pop() # if player 2 wins, pop player 1's last move from and give it a negative reward
                    dqn_solver.remember(state_copy, action_player1, reward_player2*-1, state_next_copy, terminal)
                dqn_solver.remember(normalized_state, action_player2, reward_player2, normalized_state_next, terminal)
                state = state_next

            if terminal:
#                if player == 1:
#                    player1won += 1
#                else:
#                    player2won += 1
#                try:
#                    winRatio = player1won/player2won
#                except ZeroDivisionError:
#                    winRatio = 0
#                print('Win ratio: {}'.format(winRatio)) #debug stuff
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", moves: " + str(step))
                break

        dqn_solver.experience_replay()

def evaluate_dqn(env, dqn, numberOfGames = 1000):
    print('Testing AI vs. random move. AI is yellow (Player 1)')
    tmp_exploration_rate = dqn.exploration_rate
    dqn.exploration_rate = 0
    run = 0
    aiWin = 0
    randomWin = 0
    tieCOunter = 0
    state = env.reset()

    i = 0
    while os.path.exists("Evaluation%s" % i):
        i += 1
    os.makedirs("Evaluation%s" % i)
    os.chdir("Evaluation%s" % i)

    while run < numberOfGames:
        terminal = 0
        #print('Game number {}'.format(run))
        while not terminal:
            if env.getNextPlayer() == 1: #fixed player number, because looser starts next game
                action = dqn.act(state, env)
                state_next, reward, terminal, info = env.makeMove(1, action, printflag = 0, imageflag = run < GAMES_RECORDED_PER_EVAL)
                state = state_next
                if terminal and reward > 0:
                    aiWin += 1
                if terminal and reward == 0:
                    tieCOunter += 1
            else:
                state_next, reward, terminal, info = env.makeMove(2, env.sample(), printflag = 0, imageflag = run < GAMES_RECORDED_PER_EVAL)
                state = state_next
                if terminal and reward > 0:
                    randomWin += 1
                if terminal and reward == 0:
                    tieCOunter += 1

        if run < GAMES_RECORDED_PER_EVAL:
                    env.makeVideo(run)
        run += 1

    os.chdir("..")
    print('AI won {} out of {} games. {} ties'.format(aiWin, numberOfGames, tieCOunter))
    dqn.exploration_rate = tmp_exploration_rate
    return aiWin, tieCOunter

if __name__ == "__main__":
    connect4dqn()
