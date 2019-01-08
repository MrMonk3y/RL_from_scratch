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
        
        os.chdir('to_evaluate') # has no effect on search path, but needed to save evaluations in the right place
        
        for file_name in glob.glob("*.py"):
            if file_name != '__init__.py':
                #self.solvers.append( __import__("to_evaluate.{}".format(file_name[:-3])) ) #cut away .py
                solver = getattr(import_module("to_evaluate.{}".format(file_name[:-3])), 'DQNSolver') 
                self.dqn.append(solver(self.observation_space, self.action_space))
                self.dqn[-1].exploration_rate = 0
                print('Loading weights {}.h5 ...'.format(file_name[:-3]))
                self.dqn[-1].load('{}.h5'.format(file_name[:-3]) )
                print("Added Network {}".format(file_name[:-3]))
                
        self.scores = np.zeros(len(self.dqn), dtype =int)
                    
    def get_dqn_name(self, ai):
        return self.dqn[ai].__module__.split('.')[-1] #extract moudle name
    
    def evaluate_all_dqn(self, numberOfGames = 1000, games_recorded_per_eval = 10):
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
    evaluator = TestSetup()
    evaluator.evaluate_all_dqn(numberOfGames= 50, games_recorded_per_eval = 5)
    
        
        



