import gym
import numpy as np
import nashpy as nash
from collections import defaultdict
from dqn import DQNAgent
import random
def initialize_policy(num_agents, action_space_size):
    return {agent: lambda: np.random.randint(action_space_size) for agent in range(num_agents)}

def simulate_episode(env, policies):
    obs = env.reset()
    done = [False] * env.n_agents
    while not all(done):
        actions = [policies[i]() for i in range(env.n_agents)]
        obs, rewards, done, _ = env.step(actions)
    return rewards

def compute_nash_equilibrium(meta_game_payoffs):
    equilibrium_policies = {}
    num_agents = len(meta_game_payoffs)
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            game = nash.Game(meta_game_payoffs[(i, j)], meta_game_payoffs[(j, i)])
            eqs = list(game.lemke_howson_enumeration())
            equilibrium_policies[(i, j)] = eqs
            equilibrium_policies[(j, i)] = [(eq[1], eq[0]) for eq in eqs]  # Symmetric for reverse
    return equilibrium_policies

def create_policy_from_equilibrium(equilibrium, action_space_size):
    def policy(_):
        probs = np.array(equilibrium[0])  # Assuming the first equilibrium probability distribution
        return np.random.choice(np.arange(action_space_size), p=probs)
    return policy

def psro_simulation(env, generations, episodes_per_matchup):
    num_agents = env.n_agents
    action_space_size = env.action_space.n  # Assuming uniform action space
    policies = initialize_policy(num_agents, action_space_size)

    meta_game_payoffs = defaultdict(lambda: np.zeros((2, 2)))  # Adjust size as needed

    for generation in range(generations):
        for _ in range(episodes_per_matchup):
            rewards = simulate_episode(env, policies)
            # Update meta game payoffs here based on rewards

        nash_equilibria = compute_nash_equilibrium(meta_game_payoffs)
        nash_policies = {}
        for i in range(num_agents):
            # Select a Nash policy for each agent by looking at equilibria involving them
            selected_eq = random.choice([nash_equilibria[(i, j)] for j in range(num_agents) if j != i and (i, j) in nash_equilibria])
            nash_policies[i] = create_policy_from_equilibrium(selected_eq, action_space_size)

        agents = [DQNAgent(env.observation_space[i].shape[0], action_space_size) for i in range(num_agents)]
        # Train DQN agents here using Nash policies
        for _ in range(episodes_per_matchup):
            train_agents(env, agents, nash_policies)  # Adjust this function to use nash_policies correctly

    return agents

def train_agents(env, agents, nash_policies):
    states = env.reset()
    done = [False] * env.n_agents
    while not all(done):
        actions = [nash_policies[i](states[i]) for i in range(env.n_agents)]
        next_states, rewards, done, _ = env.step(actions)
        for i, agent in enumerate(agents):
            agent.remember(states[i], actions[i], rewards[i], next_states[i], done[i])
            agent.replay()
        states = next_states

def run_computed_policies_dqn(env, agents):
    states = env.reset()
    done = [False] * env.n_agents
    while not all(done):
        actions = [agent.act(state) for agent, state in zip(agents, states)]
        states, rewards, done, _ = env.step(actions)
        print(f'Actions: {actions}, Rewards: {rewards}, Done: {done}')

env = gym.make('TrafficJunction4-v0')
agents = psro_simulation(env, 5, 100)  # Simulate 5 generations with 100 episodes per matchup
run_computed_policies_dqn(env, agents)  # Test the trained DQN agents
env.close()