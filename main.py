import gym
import ma_gym 
import numpy as np
import random
import pandas as pd
from ma_gym.envs.traffic_junction import TrafficJunction  
from collections import defaultdict
import nashpy as nash
from dqn import DQNAgent
import matplotlib.pyplot as plt


info_df = pd.DataFrame(columns=['Actions', 'Observations', 'Information', 'Rewards', 'Done'])
info_df_dqn_training = pd.DataFrame(columns=['Actions', 'Observations', 'Information', 'Rewards', 'Done'])
episodes = 2
generations = 2
max_steps = 10
runs = 2


'''
Policy Initialization Randomly
'''

def initialize_policy(num_agents, action_space_size):
    return {agent: lambda obs, size=action_space_size: np.random.randint(size) for agent in range(num_agents)}


def simulate_episode(env, policies, generation):
    if generation == 0:
        obv = env.reset()
    else:
        obv = env.get_agent_obs()
    done = [False for _ in range(env.n_agents)]

    cumulative_rewards = [0 for _ in range(env.n_agents)]

    while not all(done):
        actions = [policies[agent](obv[agent]) for agent in range(env.n_agents)]
        obv, rewards, done, info, direction, pos = env.step(actions)
        cumulative_rewards = [cumulative_rewards[i] + rewards[i] for i in range(env.n_agents)]

    return cumulative_rewards

def simulate_episode_dqn(env, agents):
    states = env.reset()
    done = [False] * env.n_agents
    cumulative_rewards = [0] * env.n_agents
    while not all(done):
        actions = [agents[i].act(states[i]) for i in range(env.n_agents)]
        next_states, rewards, done, _ = env.step(actions)
        for i in range(env.n_agents):
            cumulative_rewards[i] += rewards[i]
        states = next_states
    return cumulative_rewards

def compute_nash_equilibrium(meta_game_payoffs): #iterative method
    equilibrium_policies = {}
    for match, payoff_matrix in meta_game_payoffs.items():
        game = nash.Game(payoff_matrix[0], payoff_matrix[1])
        try:
            #if payoff_matrix[0] == payoff_matrix[1]:
                #eq = game.lemke_howson(initial_dropped_label=1)
            eq = game.lemke_howson(initial_dropped_label=0)
            equilibrium_policies[match] = [eq]
        except Exception as e:
            print(f"Failed to find Nash Equilibrium for match {match} using Lemke-Howson: {e}")
            # Fallback to support enumeration or other methods if Lemke-Howson fails
            eqs = list(game.support_enumeration())
            equilibrium_policies[match] = eqs
    return equilibrium_policies

def train_agents(env, agents, episodes=5, policies=None):
    episode_rewards = []
    for episode in range(episodes):
        states = np.array(env.reset())
        done = [False] * env.n_agents
        print(f"State Observation: {states}") 
        total_episode_reward = 0  
        while not all(done):
            actions = []
            for i in range(env.n_agents):
                if not done[i]:
                    # Incorporate Nash policies to influence action selection
                    holder = random.random()
                    nash_probabilities = policies[i](states[i]) if policies and i in policies else [0.5, 0.5]
                    print(f"Agent {i} Nash Probabilities: {nash_probabilities}")    
                    print(f"Agent {i} Acting")
                    action = agents[i].act(states[i], nash_probabilities)
                    print(f"Agent {i} Action: {action}")
                    actions.append(action)
                else:
                    actions.append(None)  # or some default no-op action if the agent is done

            # Execute actions in the environment
            for i in range(len(actions)):
                if actions[i] is None:
                    actions[i] = 0  #this happens once the agent has made it to its destination so it should stop, has no need to move.
            print(f"Actions: {actions}")
            print(f"Stepping...")
            next_states, rewards, done, _, direction, next_pos = env.step(actions)
            total_episode_reward += sum(rewards)
            episode_rewards.append(total_episode_reward)
            for i in range(len(rewards)):
                print(f"Reward {agents[i]}: {rewards[i]}")

            # Store experiences and train
            for i in range(env.n_agents):
                if not done[i]:
                    print(f"Agent {i} Remembering")
                    agents[i].remember(states[i], actions[i], rewards[i], next_states[i], done[i], direction[i], next_pos[i])
                    print(f"Agent {i} Replaying")
                    agents[i].replay()
            print(f"Done: {done}")
            states = next_states

            if all(done):
                break
    
    return episode_rewards


def create_policy_from_equilibrium(equilibrium, action_space_size):
    # This function creates a policy function from a given Nash equilibrium distribution
    def policy(observation):
        action_probabilities = equilibrium  # Assuming equilibrium is a distribution over actions
        action = np.random.choice(np.arange(action_space_size), p=action_probabilities)
        return action
    return policy


def psro_simulation(env, generations, episodes_per_matchup, flag):
    num_agents = env.n_agents
    action_space_size = 2
    agent_holder= [DQNAgent(81, action_space_size) for _ in range(num_agents)] 
    agents = {}
    total_rewards = []
    for id in range(num_agents):
        agents[id] = agent_holder[id]
    meta_game_payoffs = defaultdict(lambda: np.zeros((num_agents, action_space_size, action_space_size)))

    if flag == "Nash":
        policies = initialize_policy(num_agents, action_space_size)

        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}")

            for episode in range(episodes_per_matchup):
                for agent1 in range(num_agents):
                    for agent2 in range(num_agents):
                        cumulative_rewards = simulate_episode(env, policies, generation)
                        meta_game_payoffs[(agent1, agent2)] += np.array(cumulative_rewards).reshape((num_agents, 1, 1))
                        meta_game_payoffs[(agent2, agent1)] += np.array(cumulative_rewards).reshape((num_agents, 1, 1))
                
            #print(meta_game_payoffs)
            
            nash_equilibrium_policies = compute_nash_equilibrium(meta_game_payoffs)
            for game, payoff_matrix in nash_equilibrium_policies.items():
                print(f"Nash Game: {game}")
                print(f"Nash Payoff Matrix: {payoff_matrix}")
            
            for agent1 in range(num_agents):
                for agent2 in range(agent1 + 1, num_agents):
                    if (agent1, agent2) in nash_equilibrium_policies:
                        equilibrium = random.choice(nash_equilibrium_policies[(agent1, agent2)]) #in case of multiple equilibriums
                        policies[agent1] = create_policy_from_equilibrium(equilibrium[0], action_space_size)
                        policies[agent2] = create_policy_from_equilibrium(equilibrium[1], action_space_size)
            
            #train agents
            episode_rewards = train_agents(env, agents, episodes_per_matchup, policies)
            #plot episode rewards
            for reward in episode_rewards:  
                total_rewards.append(reward)
            print(f"reward: {episode_rewards}")

        #plot total rewards
        

        plt.plot(episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()

            #plot losses
        for agent in agents: #plot loss of each agent
            agents[agent].plot_losses()

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

def run_computed_policies_dqn(env, agents, runs):
    for i in range(runs):
        observations = env.reset()
        done = [False] * env.n_agents
        while not all(done):
            actions = [agents[i].act(observations[i]) for i in range(env.n_agents)]
            observations, rewards, done, _, direction, pos = env.step(actions)
            print(f'Actions: {actions}, Observations: {observations}, Rewards: {rewards}, Done: {done}, Direction: {direction}, Position: {pos}')
            env.render()

        

gym.envs.register(
    id='TrafficJunction4-v0',
    entry_point='ma_gym.envs.traffic_junction:TrafficJunction',
    kwargs={'max_steps': max_steps}
)
env = gym.make('TrafficJunction4-v0')
agents = psro_simulation(env, generations, 4, "Nash")
print("Running Computed Policies")
run_computed_policies_dqn(env, agents, 15) #nash 



env.close()