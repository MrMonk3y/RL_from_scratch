#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 07:56:58 2018

@author: saschajecklin
"""
import os
import glob
from connect4game import Connect4
from importlib import import_module
import evaluate
import numpy as np

class TestSetup():
    def __init__(self):
        self.env = Connect4()
        self.observation_space = self.env.reset().shape
        self.action_space = self.env.validMoves().size
        self.dqn = []
        self.dqn_name = []

        for file_name in glob.glob("v*"):
            if file_name != '__init__.py':
                self.dqn_name.append(file_name)
                solver = getattr(import_module("{}.dqn".format(file_name)), 'DQNSolver')
                self.dqn.append(solver(self.observation_space, self.action_space))
                self.dqn[-1].exploration_rate = 0
                print('Loading weights {}/weights.h5 ...'.format(file_name))
                self.dqn[-1].load('{}/weights.h5'.format(file_name) )
                print("Added Network {}".format(file_name))

        os.chdir('all_ais_evaluated') # has no effect on search path, but needed to save evaluations in the right place
        self.scores = np.zeros(len(self.dqn), dtype =int)

    def get_dqn_name(self, ai):
#        return self.dqn[ai].__module__.split('.')[-1] #extract moudle name
        return self.dqn_name[ai]

    def evaluate_all_dqn(self, numberOfGames = 10, games_recorded_per_eval = 5):
        number_of_ais = len(self.dqn)
        for i in range(number_of_ais):
            for j in range(i+1, number_of_ais):
                i_win, j_win, _ = evaluate.ai_vs_ai(self.env,
                                                 self.dqn[i], self.get_dqn_name(i),
                                                 self.dqn[j], self.get_dqn_name(j),
                                                 '', #needed to prevent adding a zero to the filename
                                                 numberOfGames, games_recorded_per_eval)
                self.scores[i] += i_win
                self.scores[j] += j_win

        print('\nTotal score:')
        for idx, x in enumerate(self.scores):
            print ('{}: {}'.format(self.get_dqn_name(idx), x) )

if __name__ == "__main__":
    print('''Every folder beginning with version* is considered as network to evaluate.
Each network needs a dqn.py file and a weights.h5 file. Consider renaming untrained networks''')
    input("Press Enter when ready...")
    evaluator = TestSetup()
    evaluator.evaluate_all_dqn(numberOfGames= 10, games_recorded_per_eval = 5)
