import gym
import ma_gym  # Ensure ma_gym is imported to register its environments
import numpy as np
import random
import pandas as pd
from ma_gym.envs.traffic_junction import TrafficJunction  
from collections import defaultdict
import nashpy as nash
from dqn import DQNAgent

info_df = pd.DataFrame(columns=['Actions', 'Observations', 'Information', 'Rewards', 'Done'])
info_df_dqn_training = pd.DataFrame(columns=['Actions', 'Observations', 'Information', 'Rewards', 'Done'])
episodes = 100
generations = 100


'''
Policy Initialization Randomly
'''

def initialize_policy(num_agents, action_space_size):
    return {agent: lambda: np.random.randint(action_space_size) for agent in range(num_agents)}



def simulate_episode(env, policies):
    obv = env.reset()
    done = [False] * env.n_agents
    cumulative_rewards = [0] * env.n_agents
    while not all(done):
        actions = [policies[agent]() for agent in range(env.n_agents)]
        obv, rewards, done, _ = env.step(actions)
        cumulative_rewards = [cum + rew for cum, rew in zip(cumulative_rewards, rewards)]
    return cumulative_rewards

def compute_nash_equilibrium(meta_game_payoffs):
    equilibrium_policies = {}
    for match, payoff_matrix in meta_game_payoffs.items():
        game = nash.Game(payoff_matrix)
        try:
            eqs = list(game.lemke_howson_enumeration())
            equilibrium_policies[match] = eqs
        except Exception as e:
            print(f"Error finding Nash Equilibrium: {e}")
    return equilibrium_policies

def train_agents(env, agents, nash_policies, episodes=10):
    for episode in range(episodes):
        states = env.reset()
        done = [False] * env.n_agents
        while not all(done):
            actions = [agents[i].act(nash_policies[i](states[i])) for i in range(env.n_agents)]  # Nash policies are used here
            next_states, rewards, done, _ = env.step(actions)
            for i, agent in enumerate(agents):
                agent.remember(states[i], actions[i], rewards[i], next_states[i], done[i])
                agent.replay()
            states = next_states
'''
def train_agents(env, agents, episodes=10):
    for episode in range(episodes):
        states = env.reset()
        done = [False] * env.n_agents
        counter = 0
        while not all(done):
            counter += 1
            print(f"Episode {episode}")
            #actions = [agent.act(state[id]) for id, agent in agents.items()]
            print(f"Agents Training Episode {episode}")
            actions = [agents[i].act(states[i]) for i in range(env.n_agents)]
            print(f"Actions: {actions}")
            print(f"Stepping...")
            obv, rewards, done2, info = env.step(actions)
            #update done with the new done values
            done = done2
            print(f"Done: {done}")
            #self play for each agent
            print(f"Self play for each agent")
            for id in range(env.n_agents):
                if done[id] == True:
                    print(f"Agent {id} has completed its episode and will no longer be trained until reset.")
                else:
                    print(f"Agent {id} not done")
                    print(f"Agents {id} reward: {rewards[id]}")
                    print(f"Agents {id} done: {done[id]}")
                    print(f"Agents {id} state: {states[id]}")
                    print(f"Agents {id} action: {actions[id]}")
                    print(f"Femember for agent {id}")
                    agents[id].remember(states[id], actions[id], rewards[id], obv[id], done[id])  # Update each agent's memory
                    print(f"Replay for agent {id}")
                    agents[id].replay()  # Train using mini-batch from memory
            
            states = obv
            
'''


def create_policy_from_equilibrium(equilibrium, action_space_size):
    def policy(_):
        print(f"Equilibrium: {equilibrium}")
        return np.random.choice(np.arange(action_space_size), p=equilibrium)
    return policy

'''
def psro_simulation(env, generations, episodes_per_matchup):
    num_agents = env.n_agents
    action_space_size = 2
    meta_game_payoffs = defaultdict(lambda: np.zeros((num_agents, action_space_size, action_space_size)))


    
    policies = initialize_policy(num_agents, action_space_size)

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        print(f"constructing meta game payoffs and policies for generation {generation + 1}")
        for episode in range(episodes_per_matchup):
            rewards = simulate_episode(env, policies)
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
        #no we have the nash equilibrium policies 
        #train dqn agents on the nash equilibrium policies
        #initialize DQN agents
        print(f"Finished constructing meta game payoffs and policies for generation {generation + 1}")
        print(f"Initializing DQN agents for generation {generation + 1}")
        agent_holder= [DQNAgent(env.observation_space[0].shape[0], action_space_size) for _ in range(num_agents)] 
        agents = {}
        for id in range(num_agents):
            agents[id] = agent_holder[id]

        #train agents
            print(f"Training agents for generation {generation + 1}")
            train_agents(env, agents, policies)  # Train all agents

            

        return agents
'''
def psro_simulation(env, generations, episodes_per_matchup):
    num_agents = env.n_agents
    action_space_size = 2  # Adjust according to actual env settings

    policies = initialize_policy(num_agents, action_space_size)
    meta_game_payoffs = defaultdict(lambda: np.zeros((num_agents, num_agents)))

    for generation in range(generations):
        print(f"Generation {generation + 1}")
        for _ in range(episodes_per_matchup):
            rewards = simulate_episode(env, policies)
            for i in range(num_agents):
                for j in range(num_agents):
                    meta_game_payoffs[i][j] += rewards[j]  # Accumulate rewards for meta-game payoff matrix

        # Normalize payoffs
        for key in meta_game_payoffs:
            meta_game_payoffs[key] /= episodes_per_matchup

        nash_equilibria = compute_nash_equilibrium(meta_game_payoffs)
        nash_policies = {}
        for i in range(num_agents):
            for j in range(num_agents):
                nash_policies = {i: create_policy_from_equilibrium(eq[0], action_space_size) for eq in nash_equilibria[(i, j)]}

        # Initialize DQN agents and train them based on Nash policies
        agents= [DQNAgent(env.observation_space[0].shape[0], action_space_size) for _ in range(num_agents)] 

        train_agents(env, agents, nash_policies, episodes_per_matchup)

    return agents
def run_computed_policies(env, policies):
    observations = env.reset()
    done = [False] * env.n_agents
    while not all(done):
        actions = [policies[agent](observations[agent]) for agent in range(env.n_agents)]
        observations, rewards, done, info = env.step(actions)
        # Ensure that each entry is recorded per step for all agents
        info_df.loc[len(info_df)] = [actions, list(observations), dict(info), list(rewards), list(done)]
        env.render()
        print(f'Actions: {actions}, Observations: {observations}, Rewards: {rewards}, Done: {done}')
    #info_df.to_csv('info.csv')

def run_computed_policies_dqn(env, agents):
    states = env.reset()
    done = [False] * env.n_agents
    while not all(done):
        actions = [agent.act(state) for agent, state in zip(agents, states)]
        states, rewards, done, _ = env.step(actions)
        print(f'Actions: {actions}, Rewards: {rewards}, Done: {done}')

        

gym.envs.register(
    id='TrafficJunction4-v0',
    entry_point='ma_gym.envs.traffic_junction:TrafficJunction',
    kwargs={'max_steps': 10}
)
env = gym.make('TrafficJunction4-v0')

agents = psro_simulation(env, 5, 100)  # Simulate 5 generations with 100 episodes per matchup
run_computed_policies_dqn(env, agents)  # Test the trained DQN agents


env.close()
