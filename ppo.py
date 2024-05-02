import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.output = nn.Linear(24, action_size)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, advantages, old_probs):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        probs = self.softmax(self.output(x))
        return probs

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.output = nn.Linear(24, 1)
        

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.output(x)
        return value

class PPOAgent:
    def __init__(self, state_size, action_size=2, gamma=0.99, clip_ratio=0.2, batch_size=40, actor_lr=0.001, critic_lr=0.001, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.memory = []
        self.losses = []
        self.epsilon = 0.01
        self.total_rewards = []
        self.reward = 0
        np.random.seed(seed)
        torch.manual_seed(seed)

    def ppo_loss(self, probs, actions, advantages, old_probs):
        m = probs.gather(1, actions.unsqueeze(1))
        old_m = old_probs.gather(1, actions.unsqueeze(1))
        ratio = torch.exp(torch.log(m) - torch.log(old_m))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
        return -torch.min(surr1, surr2).mean()

    def remember(self, state, action, prob, reward, next_state, done, direction, pos):
        self.memory.append((state[:self.state_size], action, prob, reward, next_state[:self.state_size], done, direction, pos))

    def act(self, state, nash_prob):
        state = torch.FloatTensor(state[:self.state_size]).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state, None, None)
        probabilities = probs.numpy()[0] * nash_prob
        #action = np.random.choice(self.action_size, p=probabilities)
        action = np.argmax(probabilities)
        return action, probabilities

    def act_simple(self, state):
        state = torch.FloatTensor(state[:self.state_size]).unsqueeze(0)
        with torch.no_grad():
            probabilities = self.actor(state, None, None)
        action = probabilities.argmax().item()
        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, old_probs, rewards, next_states, dones, _, _ = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_probs = torch.FloatTensor(old_probs)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        values = self.critic(states)
        next_values = self.critic(next_states)

        targets = rewards + self.gamma * (1 - dones) * next_values.squeeze()
        advantages = targets - values.squeeze()

        probs = self.actor(states, advantages, old_probs)
        actor_loss = self.ppo_loss(probs, actions, advantages, old_probs)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)  # Retain the graph here
        self.actor_optimizer.step()

        critic_loss = nn.MSELoss()(values.squeeze(), targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()  # No need to retain graph here
        self.critic_optimizer.step()

        self.losses.append(actor_loss.item())
        self.memory = []


    def plot_losses(self):
        plt.plot(self.losses)
        plt.title('Loss over Time')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.show()


