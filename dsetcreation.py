
import minari
import gymnasium as gym
from minari import DataCollector


env = gym.make('FrozenLake-v1')
env = DataCollector(env)

for _ in range(5):
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # <- use your policy here
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated

dataset = env.create_dataset("frozenlake-test-v0-2")
#dataset = minari.load_dataset("frozenlake-test-v0")
#print(dataset)
print(env.action_space.n)
for episode_data in dataset.iterate_episodes():
    #print(episode_data)
    observations = episode_data.observations
    actions = episode_data.actions
    print(actions)
    
    rewards = episode_data.rewards
    #print(rewards)
    terminations = episode_data.terminations
    truncations = episode_data.truncations
    infos = episode_data.infos