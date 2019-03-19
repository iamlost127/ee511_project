import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras.models import load_model

MAX_EPSILON_ITERS = 1000000
MIN_EPSILON = 0.1
MEM_SIZE = 1000
BATCH_SIZE = 32
IN_SAMPLES = 4

INPUT_DIM = 3*IN_SAMPLES

class flappybot:
    def __init__(self, load, train):
        # parameters
        self.epsilon = 1.0
        self.gamma = 0.99
        self.iter_count = 0
        self.episode = 1
        self.lr = 0.00025
        self.scores = []

        # Initialize memory
        self.memory = deque(maxlen=MEM_SIZE)
        for i in range(MEM_SIZE):
            rand_state = np.random.rand(1, 3)
            rand_action = 1 if random.random() > 0.5 else 0
            rand_reward = random.choice([1, -1000])
            rand_state_n = np.random.rand(1, 3)
            self.memory.append((rand_state, rand_action, rand_reward, rand_state_n))

        if load:
            print("Loading model...")
            self.model = load_model('flappybot.h5')
        else:
            print("Creating model...")
            # model
            self.model = Sequential()
            self.model.add(Dense(units=32, activation='relu', input_dim=INPUT_DIM))
            self.model.add(Dense(units=32, activation='relu'))
            self.model.add(Dense(units=2, activation='softmax'))

            # optimizer
            optimizer = optimizers.Adam(lr=self.lr)
            self.model.compile(optimizer=optimizer, loss='mse')

        self.train = train

        self.prev_state = np.random.rand(1, 3)
        self.prev_action = 1

    def remember(self, state, action, reward, state_n):
        self.memory.appendleft((state, action, reward, state_n))

    def replay(self):
        # Priorotized sampling
        batch = []
        i = 0
        while(len(batch) < BATCH_SIZE):
            if abs(self.memory[i][2]) < 100:
                sampling_prob = 0.1
            else:
                sampling_prob = abs(self.memory[i][2])/2000

            if random.random() < sampling_prob:
                batch.append(self.memory[i])

            i = (i + 1) % len(self.memory)

        for (state, action, reward, state_n) in batch:
            target = reward
            if reward == 1:
                target = reward + self.gamma * np.amax(self.predict(state_n)[0])

            target_f = self.predict(state)
            target_f[0][action] = target
            self.learn(state, target_f)

    def history(self, sample):
        inp = sample
        for i in range(IN_SAMPLES-1):
            inp = np.append(inp, self.memory[i][0])

        return np.array([inp])

    def predict(self, sample):
        return self.model.predict(self.history(sample))

    def learn(self, sample, target):
        self.model.fit(self.history(sample), target, epochs=1, verbose=0)

    def act(self, delX, delY1, vel, status, score):
        # Current state
        curr_state = np.array([[delX, delY1, vel]])
        
        # Reward for previous action
        if status:
            if delX < 0.2 and delY1 > 0.05 and delY1 < 0.35:
                reward = 1#250
            elif delY1 > 0.05 and delY1 < 0.35:
                reward = 1#50
            else:
                reward = 1
        else:
            self.episode += 1
            self.scores.append(score)
            reward = -1000


        # Epsilon-greedy strategy for first few iterations
        if self.train and random.random() < self.epsilon:
            action = 0 if random.random() < 0.7 else 1
        else:
            q_vals = self.predict(curr_state)[0]
            action = 0 if q_vals[0] > q_vals[1] else 1
            print("Q = {:03.2f} {:03.2f} | ".format(q_vals[0], q_vals[1]), \
                    "epsilon = {:07.6f} | ".format(self.epsilon), "reward = {: 5d} | ".format(reward), \
                    "score = {:4d} | ".format(score), "iter_count = {:6d} | ".format(self.iter_count), \
                    "episode = {:5d} | ".format(self.episode), "state =", curr_state)

        if self.iter_count < MAX_EPSILON_ITERS:
            self.iter_count += 1
            if self.epsilon > MIN_EPSILON:
                self.epsilon *= (1 - (self.iter_count/MAX_EPSILON_ITERS))

        if self.train:
            self.remember(self.prev_state, self.prev_action, reward, curr_state)
            self.replay()
            self.prev_state = curr_state
            self.prev_action = action

        return (action == 1)
    
    def save(self):
        self.model.save('flappybot.h5')

        with open('scores.txt', 'a') as f:
            for score in self.scores:
                f.write("%s\n" % score)
        f.close()
