# ANOTHER TEST WITH FIXED SEEDS
## Seed value
## Apparently you may use different seed values at each stage
#seed_value= 1337
#
## 1. Set `PYTHONHASHSEED` environment variable at a fixed value
#import os
#os.environ['PYTHONHASHSEED']=str(seed_value)
#
## 2. Set `python` built-in pseudo-random generator at a fixed value
#import random
#random.seed(seed_value)
#
## 3. Set `numpy` pseudo-random generator at a fixed value
#import numpy as np
#np.random.seed(seed_value)
#
## 4. Set `tensorflow` pseudo-random generator at a fixed value
#import tensorflow as tf
#tf.set_random_seed(seed_value)
#
## 5. Configure a new global `tensorflow` session
#from keras import backend as K
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

# ANOTHER TEST WITH FIXED SEEDS
#from numpy.random import seed
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)
import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam, SGD,RMSprop
from keras import regularizers
from keras.backend.tensorflow_backend import set_session

from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(7, input_shape=(observation_space,), activation="relu",kernel_regularizer=regularizers.l2(0.005)))
        #self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation="relu",kernel_regularizer=regularizers.l2(0.005)))
        self.model.add(Dense(7, activation="relu",kernel_regularizer=regularizers.l2(0.005)))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))#RMSprop(lr=0.00025, rho=0, epsilon=1e-6, decay=0.99))#

        #self.inputs = Input(shape=(observation_space,))
        #self.x = Dense(24, activation='relu')(self.inputs)
        #self.predictions = Dense(self.action_space, activation="linear")(self.x)
        #self.model = Model(inputs=self.inputs, outputs=self.predictions)
        #self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        ## This returns a tensor
        #inputs = Input(shape=(784,))

        ## a layer instance is callable on a tensor, and returns a tensor
        #x = Dense(64, activation='relu')(inputs)
        #x = Dense(64, activation='relu')(x)
        #predictions = Dense(10, activation='softmax')(x)

        ## This creates a model that includes
        ## the Input layer and three Dense layers
        #model = Model(inputs=inputs, outputs=predictions)
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole():
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
# NEEDED TO RUN CODE ONLY ON CPU ON DGX
    config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )
    sess = tf.Session(config=config)
    set_session(sess)

    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()

if __name__ == "__main__":
    cartpole()
