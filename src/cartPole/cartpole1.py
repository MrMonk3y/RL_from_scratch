#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 13:17:26 2018

@author: saschajecklin
"""

# First test with the Open AI GYM

import gym

env = gym.make('CartPole-v0')

done = False
cnt = 0

observation = env.reset()

while not done:
    #env.render()

    cnt += 1

    action = env.action_space.sample()

    observation, reward, done, _ = env.step(action)

    if done:
        break

print('game lasted ', cnt, 'moves')

env.close()
