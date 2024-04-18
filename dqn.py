import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers, Sequential
from collections import deque
import random


class DQNAgent:
    def __init__(self, state_size, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.model = self._create()
        self.gamma = .95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
    def _create(self):
        model = Sequential(
            [
                layers.Dense(24, input_shape=(self.state_size,), activation='relu'),
                layers.Dense(24, activation='relu'),
                layers.Dense(self.action_size, activation='linear')
            ]
        )
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001))
        return model
    
    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + 0.95 * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        small_batch = random.sample(self.memory, min(len(self.memory), batch_size)) 
        for state, action, reward, next_state, done in small_batch:
            target = reward
            next_state = np.array(next_state).reshape(1, -1)
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            state = np.array(state).reshape(1, -1)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        state = np.array(state).reshape(1, -1)

        if np.random.rand() <= 0.1:
            return np.random.choice(np.arange(self.action_size))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) # returns action
    

    