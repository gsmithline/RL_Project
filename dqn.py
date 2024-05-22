import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers, Sequential
from collections import deque
import random
import matplotlib.pyplot as plt

'''
Deep Q-Learning Network (DQN) Agent
'''
class DQNAgent:
    def __init__(self, state_size, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20)
        self.gamma = .99
        self.epsilon = .2
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.90
        self.learning_rate = 0.1
        self.model = self._create()
        self.model_target = self._create()
        self.losses = []
        self.history = None 
        self.action = None 
        
    def _create(self):
        model = Sequential([
                layers.Dense(24, input_shape=(self.state_size,), activation='relu'),
                layers.Dense(24, activation='relu'),
                layers.Dense(self.action_size, activation='linear')
            ])
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    
    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_state = np.array(next_state).reshape(1, -1)
            target = reward + 0.95 * np.amax(self.model.predict(next_state)[0])
        state = np.array(state).reshape(1, -1)
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=10, verbose=0)

    def remember(self, state, action, reward, next_state, done, direction, pos):
        self.memory.append((state, (action, direction, pos), reward, next_state, done))

    def replay(self, batch_size=5):
        if len(self.memory) < batch_size:
            print(f"Not enough memory to replay. Memory size: {len(self.memory)}, Required: {batch_size}")
            return

        small_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in small_batch:
            state = np.array(state, dtype=np.float32).reshape(1, self.state_size)
            next_state = np.array(next_state, dtype=np.float32).reshape(1, self.state_size)

            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                target = reward

            target_f = self.model.predict(state)
            target_f[0][action[0]] = target

            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            self.losses.append(history.history["loss"][0]) #
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state, nash_probabilities=None):
        if np.random.rand() <= self.epsilon:
            print("Random action")
            return np.random.choice(self.action_size)  
        else:
            # Adjust action probabilities based on Nash strategies if provided
            act_values = self.model.predict(state.reshape(1, -1))
            if nash_probabilities:
                act_values = act_values * nash_probabilities
                print(f"Act values: {act_values}")
            for val in act_values[0]:
                    if val != 0 or val != 1:
                        print(f"Act values: {val}")
            
            return np.argmax(act_values[0])  
        
    def plot_losses(self):
        plt.plot(self.losses)
        plt.title("Loss per Training Step")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.show()
    

    