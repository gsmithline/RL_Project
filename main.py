import gym
import ma_gym  # Ensure ma_gym is imported to register its environments
import numpy as np
import random
import pandas as pd
from ma_gym.envs.traffic_junction import TrafficJunction  
from collections import defaultdict
import nashpy as nash

info_df = pd.DataFrame()
generations = 2

'''
Policy Initialization Randomly
'''

def intialize_policy(num_agents, action_space_size):
    policies = {}
    for agent in range(num_agents):
        policies[agent] = lambda observation: np.random.randint(action_space_size)
    return policies


def simulate_episode(env, policies):

    obv = env.reset()
    done = [False for _ in range(env.n_agents)]

    cumulative_rewards = [0 for _ in range(env.n_agents)]

    while not all(done):
        actions = [policies[agent](obv[agent]) for agent in range(env.n_agents)]
        obv, rewards, done, info = env.step(actions)
        cumulative_rewards = [cumulative_rewards[i] + rewards[i] for i in range(env.n_agents)]

    return cumulative_rewards

def compute_nash_equilibrium(meta_game_payoffs):
    equilibrium_policies = {}
    for match, payoff_matrix in meta_game_payoffs.items():
        game = nash.Game(payoff_matrix[0], payoff_matrix[1])
        eqs = list(game.support_enumeration())
        equilibrium_policies[match] = eqs
    return equilibrium_policies

def create_policy_from_equilibrium(equilibrium, action_space_size):
    # This function creates a policy function from a given Nash equilibrium distribution
    def policy(observation):
        action_probabilities = equilibrium  # Assuming equilibrium is a distribution over actions
        action = np.random.choice(np.arange(action_space_size), p=action_probabilities)
        return action
    return policy


def psro_simulation(env, generations=generations, episodes_per_matchup=10):
    num_agents = env.n_agents
    action_space_size = 2 # GAS or BRAKE
    policies = intialize_policy(num_agents, action_space_size)

    #meta game payoffs
    meta_game_payoffs = defaultdict(lambda: np.zeros((num_agents, action_space_size, action_space_size)))

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")

        for episode in range(episodes_per_matchup):
            for agent1 in range(num_agents):
                for agent2 in range(agent1 + 1, num_agents):
                    # play the game
                    cumulative_rewards = simulate_episode(env, policies)
                    meta_game_payoffs[(agent1, agent2)] += np.array(cumulative_rewards).reshape((num_agents, 1, 1))
                    meta_game_payoffs[(agent2, agent1)] += np.array(cumulative_rewards).reshape((num_agents, 1, 1))

        nash_equilibrium_policies = compute_nash_equilibrium(meta_game_payoffs)

        policies = {}
        
        for agent1 in range(num_agents):
            for agent2 in range(agent1 + 1, num_agents):
                if (agent1, agent2) in nash_equilibrium_policies:
                    equilibrium = random.choice(nash_equilibrium_policies[(agent1, agent2)]) #in case of multiple equilibriums
                    policies[agent1] = create_policy_from_equilibrium(equilibrium[0], action_space_size)
                    policies[agent2] = create_policy_from_equilibrium(equilibrium[1], action_space_size)
        
        
    return policies
 
def run_computed_policies(env, policies):
    observations = env.reset()
    done = [False] * env.n_agents
    while not all(done):
        actions = [policies[agent](observations[agent]) for agent in range(env.n_agents)]
        observations, rewards, done, info = env.step(actions)
        env.render()


        

gym.envs.register(
    id='TrafficJunction10-v0',
    entry_point='ma_gym.envs.traffic_junction:TrafficJunction',
    kwargs={'max_steps': 1000}
)
env = gym.make('TrafficJunction10-v1')
policies = psro_simulation(env)
run_computed_policies(env, policies)

env.close()