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
import argparse
from score_logger import ScoreLogger
from connect4game import Connect4
from keras.backend.tensorflow_backend import set_session
from importlib import import_module

NUMBER_OF_EVAL_GAMES = 1000
SAVE_EVERY_K_GAMES = 50
GAMES_RECORDED_PER_EVAL = 10
DEMO_MODE = 0
EVAL_AI= 'v53'

def connect4dqn(folder):
    env = Connect4()
    os.chdir(folder)
    score_logger_random = ScoreLogger('AI_vs_random', average_score_to_solve=1000)
    score_logger_ai = ScoreLogger('AI_vs_{}'.format(EVAL_AI), average_score_to_solve = 11) 
    #only 10 games played but scorelogger would (early)stop(ing) when reaching 10 games 10 times in a row --> 11
    
#    player1won = 0
#    player2won = 0
    observation_space = env.reset().shape
    action_space = env.validMoves().size
    # Assign GPU to DGX
    config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )
    sess = tf.Session(config=config)
    set_session(sess)

    solver = getattr(import_module('{}.dqn'.format(folder)), 'DQNSolver')
    dqn_solver = solver(observation_space, action_space)

    run = 0
    state = env.reset() #moved one loop up. otherwise player two wont be able to start if player one wins
    while True:
        run += 1
        if run % SAVE_EVERY_K_GAMES == 0 :
            print('Saving weights and starting evaluation...')
            dqn_solver.save()
            score, ties = evaluate.ai_vs_random(env, dqn_solver, eval_ctr=run,
                                                numberOfGames = NUMBER_OF_EVAL_GAMES,
                                                games_recorded_per_eval = GAMES_RECORDED_PER_EVAL)
            score_logger_random.add_score(score + ties, run) #logging ties as success
            
            eval_solver = getattr(import_module('{}.dqn'.format(EVAL_AI)), 'DQNSolver')
            eval_dqn_solver = eval_solver(observation_space, action_space)
            
            ai1_win, ai2_win, tieCOunter = evaluate.ai_vs_ai(env, ai1=dqn_solver, ai1_name=folder,
                                                             ai2=eval_dqn_solver, ai2_name=EVAL_AI,
                                                             eval_ctr=run,
                                                             numberOfGames = 10,
                                                             games_recorded_per_eval = GAMES_RECORDED_PER_EVAL)
            del eval_dqn_solver
            score_logger_ai.add_score(ai1_win + tieCOunter, run) #logging ties as success

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

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Trains DQN')
    parser.add_argument("-f", dest="FOLDER",
                        required=True, help="Name of the Folder witch includes the DQN file")
    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args()
    connect4dqn(args.FOLDER)
