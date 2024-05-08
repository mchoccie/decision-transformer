import gym
import numpy as np

import collections
import pickle
import minari
import gymnasium as gym
from minari import DataCollector

# import d4rl


# datasets = []

# for env_name in ['halfcheetah', 'hopper', 'walker2d']:
# 	for dataset_type in ['medium', 'medium-replay', 'expert']:
# 		name = f'{env_name}-{dataset_type}-v2'
# 		env = gym.make(name)
# 		dataset = env.get_dataset()

# 		N = dataset['rewards'].shape[0]
# 		data_ = collections.defaultdict(list)

# 		use_timeouts = False
# 		if 'timeouts' in dataset:
# 			use_timeouts = True

# 		episode_step = 0
# 		paths = []
# 		for i in range(N):
# 			done_bool = bool(dataset['terminals'][i])
# 			if use_timeouts:
# 				final_timestep = dataset['timeouts'][i]
# 			else:
# 				final_timestep = (episode_step == 1000-1)
# 			for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
# 				data_[k].append(dataset[k][i])
# 			if done_bool or final_timestep:
# 				episode_step = 0
# 				episode_data = {}
# 				for k in data_:
# 					episode_data[k] = np.array(data_[k])
# 				paths.append(episode_data)
# 				data_ = collections.defaultdict(list)
# 			episode_step += 1

# 		returns = np.array([np.sum(p['rewards']) for p in paths])
# 		num_samples = np.sum([p['rewards'].shape[0] for p in paths])
# 		print(f'Number of samples collected: {num_samples}')
# 		print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

# 		with open(f'{name}.pkl', 'wb') as f:
# 			pickle.dump(paths, f)

dataset = minari.load_dataset("frozenlake-test-v0")
paths = []
for episode_data in dataset.iterate_episodes():
    ep = {}
    #print(episode_data)
    #print(dataset.action_space.n)
    observations = episode_data.observations
    observations_encoded = []
    for i in range(len(observations)):
        obs = np.array([0 for i in range(dataset.observation_space.n)])
        obs[observations[i]] = 1
        observations_encoded.append(obs)
    #print(observations_encoded)
    ep['observations'] = np.array(observations_encoded)
    actions = episode_data.actions
    action_encoded = []
    for i in range(len(actions)):
        act = np.array([0 for i in range(dataset.action_space.n)])
        act[actions[i]] = 1
        action_encoded.append(act)
    action_encoded.append([0, 0, 0, 0])
    #print(action_encoded)
    ep['actions'] = np.array(action_encoded)

    rewards = episode_data.rewards
    #ep['rewards'] = np.sum(rewards)
    #rewards.insert(0, 0)
    rewards = np.append(0, rewards)
    ep['rewards'] = np.array(rewards)
    terminations = episode_data.terminations
    #terminations.insert(0, False)
    terminations = np.append(False, terminations)
    print(terminations)
    ep['terminals'] = np.array(terminations)
    
    truncations = episode_data.truncations
    paths.append(ep)
    with open('frozenlake.pkl', 'wb') as f:
        pickle.dump(paths, f)
    # infos = episode_data.infos


