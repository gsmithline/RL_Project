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
from ppo import PPOAgent


info_df = pd.DataFrame(columns=['Actions', 'Observations', 'Information', 'Rewards', 'Done'])
info_df_dqn_training = pd.DataFrame(columns=['Actions', 'Observations', 'Information', 'Rewards', 'Done'])
episodes = 5
generations = 3
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

def train_agents(env, agents, episodes=5, policies=None, info_df=info_df, seed=42):
    episode_rewards = []
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        step_collisions = 0
        not_on_track = 0
        yield_violations = 0
        unncessary_brake_violations = 0
        efficient_crossing_violations = 0
        total_violations_cost = 0

        states = np.array(env.reset())
        done = [False] * env.n_agents
        total_episode_reward = 0  
        while not all(done):
            actions = []
            probs = []
            for i in range(env.n_agents):
                if not done[i]:
                    # Incorporate Nash policies to influence action selection
                    holder = random.random()
                    nash_probabilities = policies[i](states[i]) if policies and i in policies else [0.5, 0.5]
                    action, probilites= agents[i].act(states[i], nash_probabilities)
                    actions.append(action)
                    probs.append(probilites)
                else:
                    actions.append(None) 
                    probs.append([None, None])

            for i in range(len(actions)):
                if actions[i] is None:
                    actions[i] = 0  #this happens once the agent has made it to its destination so it should stop, has no need to move.
            next_states, rewards, done, info, direction, next_pos = env.step(actions)
            step_collisions += info['step_collisions']
            not_on_track += info['not_on_track']
            yield_violations += info['yield_violations']
            unncessary_brake_violations += info['unncessary_brake_violations']
            efficient_crossing_violations += info['efficient_crossing_violations']
            total_violations_cost += info['total_violations_cost']
            #add info to dataframe
            #info_df.loc[len(info_df)] = [seed, info['step_collisions'], info['not_on_track'], info['yield_violations'], info['unncessary_brake_violations'], info['efficient_crossing_violations'], info['total_violations_cost']]
            total_episode_reward += sum(rewards)
            episode_rewards.append(total_episode_reward)
            for i in range(env.n_agents):
                if not done[i]:
                    agents[i].remember(states[i], actions[i], probs[i], rewards[i], next_states[i], done[i], direction[i], next_pos[i])
                    agents[i].replay()
            states = next_states

            if all(done):
                break

    info_df.loc[len(info_df)] = [seed, step_collisions, not_on_track, yield_violations, 
                                    unncessary_brake_violations, efficient_crossing_violations, total_violations_cost]
    
    return episode_rewards, info_df
'''
def visualize_norm_violations(violations):

    
    columns=['seed', 'step_collisions', 'not_on_track', 'yield_violations', 
                                          'unncessary_brake_violations', 
                                          'efficient_crossing_violations', 'total_violations_cost']

    for column in columns:
        if column == 'seed':
            continue
        plt.plot(violations[column], label=column)
        plt.title(f"{column} Violations")
        plt.xlabel("Generation")
        plt.ylabel("Violations")
        plt.show()

    data = [violations[column] for column in columns if column not in ["seed", "total_violations_cost"]]
    

    plt.hist(data, bins=range(len(violations)), stacked=True, label=columns)
    plt.title("Total Norm Violations")
    plt.xlabel("Generation")
    plt.ylabel("Violations")
    plt.legend()
    plt.show()
    
'''
    


def create_policy_from_equilibrium(equilibrium, action_space_size):
    # This function creates a policy function from a given Nash equilibrium distribution
    def policy(observation):
        action_probabilities = equilibrium  # Assuming equilibrium is a distribution over actions
        action = np.random.choice(np.arange(action_space_size), p=action_probabilities)
        return action
    return policy


def psro_simulation(env, generations, episodes_per_matchup, flag, seed=42, info_training=None):
    num_agents = env.n_agents
    action_space_size = 2
    agent_holder= [PPOAgent(81, action_space_size, seed) for _ in range(num_agents)] 
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
                
            
            nash_equilibrium_policies = compute_nash_equilibrium(meta_game_payoffs)
            print(f"Solving for Nash Equilibrium in Generation {generation + 1}/{generations}") 
            print(f"Computing Nash Equilibrium for {len(nash_equilibrium_policies)} matches")
            for agent1 in range(num_agents):
                for agent2 in range(agent1 + 1, num_agents):
                    if (agent1, agent2) in nash_equilibrium_policies:
                        equilibrium = random.choice(nash_equilibrium_policies[(agent1, agent2)]) #in case of multiple equilibriums
                        policies[agent1] = create_policy_from_equilibrium(equilibrium[0], action_space_size)
                        policies[agent2] = create_policy_from_equilibrium(equilibrium[1], action_space_size)
            
            #train agents
            episode_rewards, info_df = train_agents(env, agents, episodes_per_matchup, policies, info_training, seed)
            #plot episode rewards
            for reward in episode_rewards:  
                total_rewards.append(reward)
        #plot total rewards

        return agents, total_rewards, episode_rewards, info_df
 
def run_computed_policies(env, policies):
    observations = env.reset()
    done = [False] * env.n_agents
    while not all(done):
        actions = [policies[agent](observations[agent]) for agent in range(env.n_agents)]
        observations, rewards, done, info = env.step(actions)
        # Ensure that each entry is recorded per step for all agents
        info_df.loc[len(info_df)] = [actions, list(observations), dict(info), list(rewards), list(done)]
        env.render()

def run_computed_policies_dqn(env, agents, runs, view = False, seed=42):
    rewards_avg_round = []
    results_violations_df = pd.DataFrame(columns=['seed', 'step_collisions', 'not_on_track', 'yield_violations', 
                                          'unncessary_brake_violations', 
                                          'efficient_crossing_violations', 'total_violations_cost'])
    at_dest = []
    for i in range(runs):
        steps_to_destination = 0
        observations = env.reset()
        done = [False] * env.n_agents
        steps = 0
        step_collisions = 0 
        not_on_track = 0
        yield_violations = 0
        unncessary_brake_violations = 0
        efficient_crossing_violations = 0
        total_violations_cost = 0
        while not all(done):
            steps += 1
            actions = [agents[i].act_simple(observations[i]) for i in range(env.n_agents)]
            observations, rewards, done, info, direction, pos = env.step(actions)
            step_collisions += info['step_collisions']
            not_on_track += info['not_on_track']
            yield_violations += info['yield_violations']
            unncessary_brake_violations += info['unncessary_brake_violations']
            efficient_crossing_violations += info['efficient_crossing_violations']
            total_violations_cost += info['total_violations_cost']

            rewards_avg_round.append(sum(rewards)/env.n_agents)
            steps_to_destination += 1
            if all(done):
                print(f"Episode {i} done")
            if view:
                env.render()
        if steps < max_steps:
            print(f"Episode {i} done")
            print(f"Total steps: {steps}")
        results_violations_df.loc[len(results_violations_df)] = [seed, step_collisions, not_on_track, yield_violations, 
                                    unncessary_brake_violations, efficient_crossing_violations, total_violations_cost]
        at_dest.append(steps_to_destination)
    #plot rewards
    return results_violations_df, at_dest


def simulation_10_agents(seed=42, view = False):
    info_training = pd.DataFrame(columns=['seed', 'step_collisions', 'not_on_track', 'yield_violations', 
                                          'unncessary_brake_violations', 
                                          'efficient_crossing_violations', 'total_violations_cost'])
    np.random.seed(seed)
    gym.envs.register(
    id='TrafficJunction4-v0',
    entry_point='ma_gym.envs.traffic_junction:TrafficJunction',
    kwargs={'max_steps': max_steps}
    )
    env = gym.make('TrafficJunction10-v0')
    agents, training_total_rewards, training_episode_rewards, training_norm_violations = psro_simulation(env, generations, 5, "Nash", seed, info_training)
    print("Running Computed Policies")
    sim_violated, at_dest = run_computed_policies_dqn(env, agents, 10, view) #nash 

    env.close()
    return training_total_rewards, training_episode_rewards, training_norm_violations, sim_violated, at_dest


def simulation_4_agents(seed=42, view = False):
    info_training = pd.DataFrame(columns=['seed', 'step_collisions', 'not_on_track', 'yield_violations', 
                                          'unncessary_brake_violations', 
                                          'efficient_crossing_violations', 'total_violations_cost'])
    np.random.seed(seed)
    gym.envs.register(
    id='TrafficJunction4-v0',
    entry_point='ma_gym.envs.traffic_junction:TrafficJunction',
    kwargs={'max_steps': max_steps}
    )
    env = gym.make('TrafficJunction4-v0')
    agents, training_total_rewards, training_episode_rewards, training_norm_violations = psro_simulation(env, generations, 5, "Nash", seed, info_training)
    print("Running Computed Policies")
    sim_violated, at_dest = run_computed_policies_dqn(env, agents, 10, view) #nash 

    env.close()
    return training_total_rewards, training_episode_rewards, training_norm_violations, sim_violated, at_dest



def simulation_10_to_4_agents(seed=42, view = False):
    info_training = pd.DataFrame(columns=['seed', 'step_collisions', 'not_on_track', 'yield_violations', 
                                          'unncessary_brake_violations', 
                                          'efficient_crossing_violations', 'total_violations_cost'])
    np.random.seed(seed)
    gym.envs.register(
    id='TrafficJunction4-v0',
    entry_point='ma_gym.envs.traffic_junction:TrafficJunction',
    kwargs={'max_steps': max_steps}
    )
    env = gym.make('TrafficJunction10-v0')
    agents, training_total_rewards, training_episode_rewards, training_norm_violations = psro_simulation(env, generations, 5, "Nash", seed, info_training)
    
    env.close()
    print("Running Computed Policies")
    env = gym.make('TrafficJunction4-v0')
    new_agents = {}
    #sort agents by agent.total_rewards averge
    agents = {k: v for k, v in sorted(agents.items(), key=lambda item: np.mean(item[1].total_rewards), reverse=True)}
    counter = 0
    for key, value in agents.items():
        if counter < 4:
            new_agents[counter] = value
            counter += 1
        else:
            break
    sim_violated, at_dest = run_computed_policies_dqn(env, new_agents, 10, view) #nash 
    print(sim_violated)
    env.close()
    return training_total_rewards, training_episode_rewards, training_norm_violations, sim_violated, at_dest


