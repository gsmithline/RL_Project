import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
class PPOAgent:
    def __init__(self, state_size, action_size=2, gamma=0.99, clip_ratio=0.2, batch_size=64, actor_lr=0.001, critic_lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.actor = self.build_actor(actor_lr)
        self.critic = self.build_critic(critic_lr)
        self.memory = []  # Memory to store trajectories for updating
        self.losses = []  # To store losses for plotting 
        self.epsilon = 0.01 
        
    def build_actor(self, learning_rate):
        input = Input(shape=(self.state_size,))
        advantages = Input(shape=(1,))
        old_prb = Input(shape=(self.action_size,))

        x = Dense(24, activation='relu')(input)
        x = Dense(24, activation='relu')(x)
        probs = Dense(self.action_size, activation='softmax')(x)

        model = Model(inputs=[input, advantages, old_prb], outputs=[probs])
        model.compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), loss=self.ppo_loss(advantages, old_prb))
        return model
    
    def build_critic(self, learning_rate):
        input = Input(shape=(self.state_size,))
        x = Dense(24, activation='relu')(input)
        x = Dense(24, activation='relu')(x)
        value = Dense(1)(x)

        model = Model(inputs=[input], outputs=[value])
        model.compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), loss='mse')
        return model
    
    def ppo_loss(self, advantages, old_prb):
        def loss(y_true, y_pred):
            prob_ratio = tf.reduce_sum(y_true * y_pred, axis=1) / tf.reduce_sum(old_prb * y_true, axis=1)
            clipped = tf.clip_by_value(prob_ratio, 1-self.clip_ratio, 1+self.clip_ratio)
            return -tf.reduce_mean(tf.minimum(prob_ratio * advantages, clipped * advantages))
        
        return loss
    
    def remember(self, state, action, prob, reward, next_state, done, direction, pos):
        self.memory.append([state, action, prob, reward, next_state, done, direction, pos])
    
    def act(self, state, nash_prob):
        rand = np.random.rand()
        if rand <= self.epsilon:
            print("Random action")

            return np.random.choice(self.action_size), [rand, 1-rand]
        else: 
            state = state.reshape(1, self.state_size)
            probabilities = self.actor.predict([state, np.zeros((1, 1)), np.zeros((1, self.action_size))])
            probabilities = nash_prob * probabilities
            #action = np.random.choice(self.action_size, p=probabilities[0])
            action = np.argmax(probabilities[0])
            return action, probabilities[0]
    def act_simple(self, state):
        state = state.reshape(1, self.state_size)
        probabilities = self.actor.predict([state, np.zeros((1, 1)), np.zeros((1, self.action_size))])
        action = np.argmax(probabilities[0])
        return action
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            print("Not enough samples in memory to perform training")
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, old_probs, rewards, next_states, dones, _, _= zip(*self.memory)

        # Convert to numpy arrays and ensure all are correctly shaped
        states = np.array(states)
        actions = np.array(actions)
        old_probs = np.array(old_probs)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Ensure old_probs is correctly shaped
        if old_probs.shape[1] != self.action_size:
            raise ValueError(f"Shape of old_probs should be ({self.batch_size}, {self.action_size}), but got {old_probs.shape}")

        # Update critic
        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)
        targets = rewards + self.gamma * (1 - dones) * next_values.squeeze()
        advantages = targets - values.squeeze()

        # Clip advantages
        advantages = np.clip(advantages, -1, 1).reshape(-1, 1)

        # Convert actions to one-hot encoding
        actions_one_hot = tf.keras.utils.to_categorical(actions, num_classes=self.action_size)

        # Train the actor model
        actor_history = self.actor.fit([states, advantages, old_probs], actions_one_hot, batch_size=self.batch_size, verbose=0)
        actor_loss = self.ppo_loss(advantages, old_probs)
        self.losses.append(actor_loss)

        critic_history = self.critic.fit(states, targets, batch_size=self.batch_size, verbose=0)
        critic_loss = self.ppo_loss(targets, values)
        self.losses.append(critic_loss)
        # Clear memory
        self.memory.clear()






    def plot_losses(self):
        plt.plot(self.losses)
        plt.title('Loss over Time')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.show()