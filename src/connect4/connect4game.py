# Author: Sascha Jecklin
# Date: 11/20/2018 15:22
# Connect Four Game in Python for Reinforcement Learning Project Thesis

import numpy as np
import sys
from scipy.ndimage.filters import convolve
from random import choice
from matplotlib import pyplot as plt
import subprocess
import glob
import os

class Connect4():
    def __init__(self, rowSize = 6, colSize= 7, winReward = 100, loseReward=-100, reward = 1):
        self.rowSize = rowSize
        self.colSize = colSize
        self.loseReward = loseReward
        self.winReward = winReward
        self.reward = reward
        self.field = np.zeros(shape=(self.rowSize, self.colSize), dtype=int)
        self.turnCounter = 0
        self.removeOldImages()
        self.removeOldVideos()

    def reset(self):
        self.field = np.zeros(shape=(self.rowSize, self.colSize), dtype=int)
        self.turnCounter = 0
        return self.stateSplitter()

    def getNextPlayer(self):
        if self.turnCounter % 2 == 0:
            return 1
        else:
            return 2

    def makeMove(self, player, col, printflag = 0, imageflag = 0): # returns -1 for error else observation, reward, done, info
        assert self.turnCounter % 2 + 1 == player, "Not your turn Player: %r" % player
        selectedRow = self.field[:, col] #row to check
        if (selectedRow==0).any() == 0: # if full return info
            sys.exit('tried to fill an already full column')
        nextEmptySpace = (selectedRow!=0).argmin() #first nonzero entry starting from the bottom
        self.field[nextEmptySpace, col] = player
        self.turnCounter += 1
        if printflag == 1:
            self.printField()
            
        if imageflag == 1:
            self.makeImage(self.turnCounter)
        reward = 0
        done = False
        info = None
        observation = self.stateSplitter()
        winner = self.checkWin() # check winner every played move
        if winner != 0:
            #self.makeVideo(self.turnCounter)
            if winner == 3:
                if printflag == 1:
                    print("It's a tie!")
                self.reset()
                done = True
                return observation, reward, done, info
            else:
                if printflag == 1:
                    print("Player {} wins".format(winner))
                self.reset()
                if winner == 1: # looser starts next round
                    self.turnCounter = 1
                reward = 1
                done = True
        return observation, reward, done, info
    
    def stateSplitter(self): # makes a matrix for player 1 and one for palyer 2
        copy_player_1 = np.copy(self.field)
        copy_player_2 = np.copy(self.field)
        copy_player_1[copy_player_1 == 2] = 0
        copy_player_2[copy_player_2 == 1] = 0
        copy_player_2[copy_player_2 == 2] = 1
        return np.stack([copy_player_1, copy_player_2], axis = -1)
        
    def printField(self):
        for i in range(np.size(self.field, 0)-1,-1,-1):
            for j in range(0, self.rowSize+1):
                if self.field[i,j] == 1:
                    print('\x1b[0;30;43m' + str(self.field[i,j]) + '\x1b[0m', end='  ')
                elif self.field[i,j] == 2:
                    print('\x1b[0;30;41m' + str(self.field[i,j]) + '\x1b[0m', end='  ') #1;30;43m in spyder
                else:
                    print(self.field[i,j], end='  ')
            print('\n')
        print("\n")
        
        
    def checkWin(self): # Convolves pattern with field. If a 4 appears --> winner
        mask = np.zeros(shape=(self.rowSize, self.colSize), dtype=int)
        if self.turnCounter % 2:
             mask[self.field==1]=1
             possbileWinner = 1
        else:
             mask[self.field==2]=1
             possbileWinner = 2
        if self.turnCounter == self.colSize * self.rowSize:
            return 3
        k1 = np.array([[1],[1],[1],[1]])
        k2 = np.array([[1,1,1,1]])
        k3 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        k4 = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
        
        convVertical = convolve(mask, k1, mode='constant', cval=0)
        if 4 in convVertical:
            return possbileWinner
        convVertical = convolve(mask, k2, mode='constant', cval=0)
        if 4 in convVertical:
            return possbileWinner
        convVertical = convolve(mask, k3, mode='constant', cval=0)
        if 4 in convVertical:
            return possbileWinner
        convVertical = convolve(mask, k4, mode='constant', cval=0)
        if 4 in convVertical:
            return possbileWinner
        
        return 0

    def validMoves(self): #returns an array colSize long with ones for validMoves
        validMovesArray = np.ones(self.colSize,dtype = int)
        for i in range(0, self.rowSize+1):
            selectedRow = self.field[:, i] #row to check
            if (selectedRow==0).any() == 0: # if full return -1
                validMovesArray[i] = 0
        return validMovesArray
    
    def sample(self): #returns a valid sample move 
        return choice(np.where(self.validMoves())[0]) # [0] to get it out of the tuple
    
    def makeImage(self, i): #makes an image from the current state
        data = np.ones((self.rowSize, self.colSize, 3), dtype=np.uint8)
        #data = data*255 #white background
        data[self.field == 1] = [255, 240, 0]
        data[self.field == 2] = [255, 0, 0]
        data = np.flip(data, 0)
        plt.figure()
        im = plt.imshow(data, interpolation='none', vmin=0, vmax=1, aspect='equal');
        ax = plt.gca();
        ax.set_xticks(np.arange(-.5, self.colSize, 1), minor=True); #need to shift grid to right position
        ax.set_yticks(np.arange(-.5, self.rowSize, 1), minor=True);
        plt.tick_params(
            axis='both',       # changes apply to the x and y-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,
            left=False,
            right=False,       # ticks along the top edge are off
            labelleft=False,
            labelbottom=False) # labels along the bottom edge are off

        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)    
        plt.savefig("move%02d.png" % i, bbox_inches='tight', pad_inches = 0)
        plt.close()
        
    def makeVideo(self, i): #makes a video out of saved images
        #print("Saving Video...")
        FNULL = open(os.devnull, 'w') #subprocess ouput redirected to devnull
        subprocess.call([
            'ffmpeg', '-framerate', '2', '-i', 'move%02d.png', '-r', '2', '-pix_fmt', 'yuv420p',
            "game%02d.mp4" % i
        ], stdout=FNULL, stderr=subprocess.STDOUT)
        self.removeOldImages()
    
    def removeOldImages(self):
        for file_name in glob.glob("*.png"):
            os.remove(file_name)
            
    def removeOldVideos(self):
        for file_name in glob.glob("*.mp4"):
            os.remove(file_name)

if __name__ == "__main__":
    game = Connect4()
    print("Player {}'s turn".format(game.getNextPlayer() ))
    
    while 1:
        info = 1
        while info != None:
            print(game.validMoves())
            userInput = int(input("Which row Player 1? "))
            observation, reward, done, info = game.makeMove(1,userInput, 1)
            if info != None:
                print(info)
        info = 1
        while info != None:
            print(game.validMoves())
            userInput = int(input("Which row Player 2? "))
            observation, reward, done, info = game.makeMove(2,userInput, 1)
            if info != None:
                print(info)

