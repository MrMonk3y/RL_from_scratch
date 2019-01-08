#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 11:20:20 2019

@author: saschajecklin
"""
import os
import numpy as np
import tensorflow as tf
import evaluate
from score_logger import ScoreLogger
from connect4game import Connect4
from keras.backend.tensorflow_backend import set_session
from version00.dqn import DQNSolver

NUMBER_OF_EVAL_GAMES = 5
SAVE_EVERY_K_GAMES = 12
GAMES_RECORDED_PER_EVAL = 2
DEMO_MODE = 0

def connect4dqn():
    env = Connect4()
    os.chdir('version00')
    score_logger = ScoreLogger('Connect4')
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
        if run % SAVE_EVERY_K_GAMES == 0 :
            print('Saving weights and starting evaluation...')
            dqn_solver.save()
            score, ties = evaluate.ai_vs_random(env, dqn_solver,  numberOfGames = NUMBER_OF_EVAL_GAMES , games_recorded_per_eval = GAMES_RECORDED_PER_EVAL )
            score_logger.add_score(score + ties, run) #logging ties as success
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
            
if __name__ == "__main__":
    connect4dqn()