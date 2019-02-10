# RL_from_scratch

My playground with Reinforcement Learning. The focus was on teaching an AI how to play the game Connect4 from scratch. No additional information, rules or guidelines are given to the AI. Also for simplicity's sake, rather simple networks were chosen. 

# cartPole

Very simple test with OpenAI-Gym's Environment CartPole can be found in this folder. The code is no longer mainted.

# connect4

In the connect4 folder multiple models are listed. Most of them are trained. 

Prerequisites
------------

    tensorflow
    openai-gym
    
Usage
------------

    python connect4ai.py -f versionXX
    
Trains a model. Caution! This overwrites an exisitng weight.h5 file

    python evaluate.py -f versionXX
    
Opens the console and lets a human play against the currenct state of the model

    evaluate_all.py
    
Evaluates all models in a "versionXX" folder against each other. Evaluation videos will be stored in the Folder all_ais_evaluated. Caution! Older evaluations will be overwritten. Evaluation can only be executed if model a called dqn.py and a corresponding weight.h5 file exist. 

