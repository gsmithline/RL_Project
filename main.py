import gym
import ma_gym  # Ensure ma_gym is imported to register its environments
import numpy as np
import random
from ma_gym.envs.traffic_junction import TrafficJunction  


# Register the custom environment
gym.envs.register(
    id='TrafficJunction10-v0',
    entry_point='ma_gym.envs.traffic_junction:TrafficJunction',
    kwargs={'max_steps': 1000}
)

# Initialize the environment
env = gym.make('TrafficJunction10-v1')

# Reset the environment to start a new episode
observation = env.reset()

'''
Policy Initialization Randomly
'''

def intialize_policy(num_agents, action_space_size):
    policies = {}

    for agent in range(num_agents):
        def policy(observation):
            return np.random.randint(action_space_size)
        policies[agent] = policy

    return policies
    

num_agents = env.n_agents
action_space_size = 2 # GAS or BRAKE
policies = intialize_policy(num_agents, action_space_size)
observation = None

for key in policies.keys():
    print(key, policies[key](observation))



'''

# Loop through steps until the episode is done
done = [False] * env.n_agents
while not all(done):
    # Randomly sample actions for each agent
    #actions = [random.choice([0, 1]) for _ in range(env.n_agents)]
    
    actions = [env.action_space.sample()]
    observation = env.observation_space.sample()
    #actions = heuristic_action(observation)
    
    
    for action in actions:
        # Take a step in the environment with the sampled actions
        observation, rewards, done, info = env.step(action)
        # Optionally, render the environment's state (if rendering is supported)
        env.render()

# Close the environment
'''
env.close()