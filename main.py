from model import ReplayBuffer
from model import D3QL
from matplotlib import pyplot as plt
from GridNavigationEnv import GridNavigationEnv
import config as cfg
import numpy as np
from copy import copy
import utils



env = GridNavigationEnv(L=cfg.L, P=cfg.P, N=cfg.N, T=cfg.T, M=cfg.M)
observation = env.reset()

dimension = (2 * cfg.M) + 1
buffer = ReplayBuffer(cfg.memory_size, cfg.N, dimension)
d3ql_algorithm = D3QL(cfg.N, dimension)

is_successful = np.zeros(cfg.episode_num)
average_rewards = np.zeros(cfg.episode_num)
distance_episodes = np.zeros(cfg.episode_num)
rate_distance_over_step = np.zeros(cfg.episode_num)

epsilon = 1
epsilon_decay = 0.99
epsilon_min = 0.05
epsilon_history = []

for ep in range(cfg.episode_num):
    done = False
    terminate = False
    observation = env.reset()
    print(f'starting episode {ep+1}')
    rewards_this_episode = []
    
    while not done and not terminate:
        old_observation = copy(observation)

        # epsilon-greedy algorithm
        if np.random.random() < epsilon:
            actions = np.array([env.action_space.sample() for _ in range(env.N)])  # Random actions
        else:
            actions = np.array([d3ql_algorithm.get_model_output(observation[i, :, :].flatten(), i)
                        for i in range(env.N)])  # Intelligent actions

        observation, rewards, terminate, info, done = env.step(actions)
        rewards_this_episode.append(rewards.mean())       

        actions_encoded = np.array([utils.return_one_hot_vector(a) for a in actions])
        buffer.store_experience(old_observation.reshape(cfg.N, -1),
                                observation.reshape(cfg.N, -1),
                                actions_encoded, rewards, done or terminate)

        # update epsilon
        epsilon *= epsilon_decay
        epsilon = max(epsilon, epsilon_min)
        epsilon_history.append(epsilon)

        # train models independently
        if buffer.mem_counter > cfg.batch_size:
            state, next_state, action, reward, dones = buffer.sample_buffer()
            d3ql_algorithm.train_independent(state, next_state, action, reward, dones)

    is_successful[ep] = done
    average_rewards[ep] = np.array(rewards_this_episode).mean() if len(rewards_this_episode) > 0 else 0
    distance_all_value = [value for sublist in info['distances'].values() for value in sublist]
    distance_episodes[ep] = sum(distance_all_value) / len(distance_all_value)
    rate_distance_over_step[ep] = distance_episodes[ep] / info['steps_number']


print('*****************************************')
print(f'Success rate for {cfg.episode_num} episodes was {is_successful.mean() * 100}%')
print(f'Average reward for {cfg.episode_num} episodes was {round(average_rewards.mean(), 3)}')
print('*****************************************')


plt.figure('Reward')
if len(average_rewards) < cfg.episode_num:
    average_rewards = np.pad(average_rewards, (0, cfg.episode_num - len(average_rewards)), 'constant', constant_values=np.nan)
plt.plot(range(1, cfg.episode_num + 1), average_rewards, marker='.')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Average Rewards Over Episodes')
plt.grid(True)
plt.savefig('Rewards_over_Episodes.png')

plt.figure('Distance')
if len(average_rewards) < cfg.episode_num:
    average_rewards = np.pad(average_rewards, (0, cfg.episode_num - len(average_rewards)), 'constant', constant_values=np.nan)
plt.plot(range(1, cfg.episode_num + 1), distance_episodes, marker='.')
plt.xlabel('Episode')
plt.ylabel('Distance')
plt.title('Average Distances Over Episodes')
plt.grid(True)
plt.savefig('Distances_over_Episodes.png')


plt.figure('Epsilon')
plt.plot(range(1, len(epsilon_history) + 1), epsilon_history, marker='.')
plt.xlabel('Update Times')
plt.ylabel('Epsilon')
plt.title('Epsilon Trend')
plt.grid(True)
plt.savefig('Epsilon_Trend.png')




plt.figure('Rate')
if len(rate_distance_over_step) < cfg.episode_num:
    rate_distance_over_step = np.pad(rate_distance_over_step, (0, cfg.episode_num - len(rate_distance_over_step)), 'constant', constant_values=np.nan)
plt.plot(range(1, cfg.episode_num + 1), rate_distance_over_step, marker='.')
plt.xlabel('Episode')
plt.ylabel('distance/step_num')
plt.title('Rate: Distance / Step_num')
plt.grid(True)
plt.savefig('Rate.png')


plt.show()