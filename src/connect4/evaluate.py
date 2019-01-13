#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 12:32:27 2019

@author: saschajecklin
"""
import os
import sys
import argparse
from connect4game import Connect4
from importlib import import_module

def ai_vs_random(env, dqn, eval_ctr=0, numberOfGames = 1000, games_recorded_per_eval = 10):
    print('Testing AI vs. random move. AI is yellow (Player 1)')
    tmp_exploration_rate = dqn.exploration_rate # saving exploration rate
    dqn.exploration_rate = 0 # setting exploration to zero while evaluating
    run = 0
    aiWin = 0
    randomWin = 0
    tieCOunter = 0
    state = env.reset()

    while os.path.exists("Evaluation{}".format(eval_ctr)):
        print("Old evaluations exits. Move them first!")
        input("Press Enter to continue...")
    os.makedirs("Evaluation{}".format(eval_ctr))
    os.chdir("Evaluation{}".format(eval_ctr))

    while run < numberOfGames:
        terminal = 0
        #print('Game number {}'.format(run))
        while not terminal:
            if env.getNextPlayer() == 1: #fixed player number, because looser starts next game
                action = dqn.act(state, env)
                state_next, reward, terminal, info = env.makeMove(1, action, printflag = 0, imageflag = run < games_recorded_per_eval)
                state = state_next
                if terminal and reward > 0:
                    aiWin += 1
                if terminal and reward == 0:
                    tieCOunter += 1
            else:
                state_next, reward, terminal, info = env.makeMove(2, env.sample(), printflag = 0, imageflag = run < games_recorded_per_eval)
                state = state_next
                if terminal and reward > 0:
                    randomWin += 1
                if terminal and reward == 0:
                    tieCOunter += 1

        if run < games_recorded_per_eval:
            env.makeVideo(run)
        run += 1

    os.chdir("..")
    print('AI won {} out of {} games. {} ties'.format(aiWin, numberOfGames, tieCOunter))
    dqn.exploration_rate = tmp_exploration_rate
    return aiWin, tieCOunter

def ai_vs_ai(env, ai1, ai1_name, ai2, ai2_name, eval_ctr=0, numberOfGames = 1000, games_recorded_per_eval = 10):

    print('Testing AI {} vs. AI {}. AI {} is yellow'.format(ai1_name, ai2_name, ai1_name))
    run = 0
    ai1_win = 0
    ai2_win = 0
    tieCOunter = 0
    state = env.reset()

    while os.path.exists("Evaluation{}_{}_vs_{}".format(eval_ctr, ai1_name, ai2_name)):
        print("Old evaluations exits. Move them first!")
        input("Press Enter to continue...")

    os.makedirs("Evaluation{}_{}_vs_{}".format(eval_ctr, ai1_name, ai2_name))
    os.chdir("Evaluation{}_{}_vs_{}".format(eval_ctr, ai1_name, ai2_name))

    while run < numberOfGames:
        terminal = 0
        #print('Game number {}'.format(run))
        while not terminal:
            if env.getNextPlayer() == 1: #fixed player number, because looser starts next game
                action = ai1.act(state, env)
                state_next, reward, terminal, info = env.makeMove(1, action, printflag = 0, imageflag = run < games_recorded_per_eval)
                state = state_next
                if terminal and reward > 0:
                    ai1_win += 1
                if terminal and reward == 0:
                    tieCOunter += 1
            else:
                action = ai2.act(state, env)
                state_next, reward, terminal, info = env.makeMove(2, action, printflag = 0, imageflag = run < games_recorded_per_eval)
                state = state_next
                if terminal and reward > 0:
                    ai2_win += 1
                if terminal and reward == 0:
                    tieCOunter += 1

        if run < games_recorded_per_eval:
            env.makeVideo(run)
        run += 1

    os.chdir("..")
    print('AI {} scored against AI {} {}:{}. {} ties'.format(ai1_name, ai2_name, ai1_win, ai2_win, tieCOunter))

    return ai1_win, ai2_win, tieCOunter

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Testing DQN against Human')
    parser.add_argument("-f", dest="FOLDER",
                        required=True, help="Name of the Folder witch includes the DQN and the corresponding weights.h5 file")
    return parser.parse_args(args)

def ai_vs_human(version_name):

    env = Connect4()
    observation_space = env.reset().shape
    action_space = env.validMoves().size

    try:
        solver = getattr(import_module("{}.dqn".format(version_name)), 'DQNSolver')
    except ImportError:
        sys.exit("There is no folder with this name that contains a working DQN for Connect4")

    dqn = solver(observation_space, action_space)
    dqn.exploration_rate = 0
    print('Loading weights...')
    dqn.load('{}/weights.h5'.format(version_name))

    print('Testing AI vs. human. AI is yellow (Player 1)')
    print (quit)

    run = 0
    aiWin = 0
    humanWin = 0
    tieCOunter = 0
    state = env.reset()

    while True:
        terminal = 0
        print('Game number {}'.format(run))
        while not terminal:
            if env.getNextPlayer() == 1: #fixed player number, because looser starts next game
                action = dqn.act(state, env)
                state_next, reward, terminal, info = env.makeMove(1, action, printflag=1)
                state = state_next
                if terminal and reward > 0:
                    aiWin += 1
                    break
                if terminal and reward == 0:
                    tieCOunter += 1
                    break
            else:
                print(env.validMoves())
                userInput = int(input("Which row silly Human? "))
                state_next, reward, terminal, info = env.makeMove(2, userInput, printflag=1)
                state = state_next
                if terminal and reward > 0:
                    humanWin += 1
                    break
                if terminal and reward == 0:
                    tieCOunter += 1
                    break
        run += 1
        if terminal:
            print('AI won {} out of {} games\n'.format(aiWin, run))

if __name__ == "__main__":
    args = parse_args()
    ai_vs_human(args.FOLDER)
