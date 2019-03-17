import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model

class flappybot:
    model = None
    prev_state = None

    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(units=8, activation='relu', input_dim=4))
        self.model.add(Dense(units=16, activation='relu'))
        self.model.add(Dense(units=2, activation='softmax'))
        self.model.compile(optimizer='sgd', loss='mse')

        self.prev_state = np.array([[0.5, 0.5, 0.5, 0.5]])
        self.model.fit(self.prev_state, np.array([[1, 0]]))

        plot_model(self.model, to_file='flappynn.png', show_shapes=True)

    def act(self, delX, delY1, delY2, vel, status):
        delX_norm = delX/450
        delY1_norm = delY1/250
        delY2_norm = delY2/250
        vel_norm = vel/10
        curr_state = np.array([[delX_norm, delY1_norm, delY2_norm, vel_norm]])

        reward = 0.001 if status else -1
        if not status: print("crashed")

        q_vals = self.model.predict(curr_state)[0]
        action = 1 if q_vals[1] > q_vals[0] else 0 # 1=flap; 0=don't flap

        prev_q_vals = self.model.predict(self.prev_state)[0]
        prev_action = 1 if prev_q_vals[1] > prev_q_vals[0] else 0

        pred_q_vals = np.array([reward, reward]) + 0.1*(q_vals)
        self.model.fit(self.prev_state, np.array([pred_q_vals]))

        prev_state = curr_state
         
        return (action == 1)
