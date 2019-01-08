#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 13:46:06 2018

@author: saschajecklin
"""

# Second test with the Open AI GYM

import gym
import numpy as np
from gym import wrappers

env = gym.make('CartPole-v0')

bestLenght = 0
episodeLenght = []

bestWeights = np.zeros(4)
# make 100 iterations and calculate an average length. If average length better then best --> update. repeat 100 times
for i in range(100):
    newWeights = np.random.uniform(-1.0, 1.0, 4)

    length = []
    for j in range(100):
        observation = env.reset()
        done = False
        cnt = 0

        while not done:
            #env.render()
            cnt += 1
            action = 1 if np.dot(observation, newWeights) > 0 else 0
            observation, reward, done, _ = env.step(action)

            if done:
                break
        length.append(cnt)
    averageLenght = float(sum(length)/len(length))

    if averageLenght > bestLenght:
        bestLenght = averageLenght
        bestWeights = newWeights

    episodeLenght.append(averageLenght)
    if i % 10 == 0 :
        print('best lenght is ', bestLenght)

#final test with best weights
envWrapper = wrappers.Monitor(env, 'CartPole2_Movie', force=True)
observation = envWrapper.reset()
done = False
cnt = 0

while not done:
    observations = envWrapper.render()
    cnt += 1

    action = 1 if np.dot(observation, bestWeights) > 0 else 0
    observation, reward, done, _ = envWrapper.step(action)

    if done:
        break

print('game lasted ', cnt, 'moves')

envWrapper.env.close()
