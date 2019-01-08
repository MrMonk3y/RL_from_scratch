#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 07:49:33 2018

@author: saschajecklin
"""
from Connect4 import Connect4
from myDQNLearningConnect4 import DQNSolver

class TestSetup():
    def __init__(self, mode = 'avr', numberOfGames = 10):
        self.mode = mode
        self.numberOfGames = numberOfGames
        self.env = Connect4()
        observation_space = self.env.reset().shape
        action_space = self.env.validMoves().size
        self.dqn = DQNSolver(observation_space, action_space )
        self.dqn.exploration_rate = 0
        print('loading weights...')
        self.dqn.load()
        
        if mode == 'avr':
            self.avr()
        elif mode == 'avh':
            self.avh()

    def avr(self):
        print('Testing AI vs. random move. AI is red (Player 1)')
        run = 0
        aiWin = 0
        randomWin = 0
        tieCOunter = 0
        state = self.env.reset()
        
        while run < self.numberOfGames:
            terminal = 0
            print('Game number {}'.format(run))
            while not terminal:
                if self.env.getNextPlayer() == 1: #fixed player number, because looser starts next game
                    action = self.dqn.act(state, self.env)
                    state_next, reward, terminal, info = self.env.makeMove(1, action)
                    state = state_next
                    if terminal and reward > 0:
                        aiWin += 1
                        break
                    if terminal and reward == 0:
                        tieCOunter += 1
                        break
                else:
                    state_next, reward, terminal, info =self.env.makeMove(2, self.env.sample())
                    state = state_next
                    if terminal and reward > 0:
                        randomWin += 1
                        break
                    if terminal and reward == 0:
                        tieCOunter += 1
                        break
                
            run += 1
        print('AI won {} out of {} games'.format(aiWin, self.numberOfGames))
        
    def avh(self):
        print('Testing AI vs. human. AI is red (Player 1)')
        run = 0
        aiWin = 0
        humanWin = 0
        tieCOunter = 0
        state = self.env.reset()
        
        while run < self.numberOfGames:
            terminal = 0
            print('Game number {}'.format(run))
            while not terminal:
                if self.env.getNextPlayer() == 1: #fixed player number, because looser starts next game
                    action = self.dqn.act(state, self.env)
                    state_next, reward, terminal, info = self.env.makeMove(1, action)
                    state = state_next
                    if terminal and reward > 0:
                        aiWin += 1
                        break
                    if terminal and reward == 0:
                        tieCOunter += 1
                        break
                else:
                    print(self.env.validMoves())
                    userInput = int(input("Which row silly Human? "))
                    state_next, reward, terminal, info =self.env.makeMove(2, userInput)
                    state = state_next
                    if terminal and reward > 0:
                        humanWin += 1
                        break
                    if terminal and reward == 0:
                        tieCOunter += 1
                        break    
            run += 1
        print('AI won {} out of {} games'.format(aiWin, self.numberOfGames))
            
if __name__ == "__main__":
    mode = 'avh'
    numberOfGames = 1000
    TestSetup(mode, numberOfGames)
    
        
        



