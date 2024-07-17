import csv
import os
from copy import copy

import numpy as np
import tqdm
from matplotlib import pyplot as plt

import config as cfg
import utils
from GridNavigationEnv import GridNavigationEnv
from model import D3QL
from model import ReplayBuffer

env = GridNavigationEnv()
observation = env.reset()

fov_size = (2 * cfg.M) + 1
dimension = (1, fov_size, fov_size)
buffer = ReplayBuffer(cfg.memory_size, cfg.N, dimension)
d3ql_algorithm = D3QL(cfg.N, dimension)

is_successful = np.zeros(cfg.episode_num)
average_rewards = np.zeros(cfg.episode_num)
distance_episodes = np.zeros(cfg.episode_num)
rate_distance_over_step = np.zeros(cfg.episode_num)

epsilon = cfg.learning_hyperparameters['epsilon']
epsilon_decay = cfg.learning_hyperparameters['epsilon_decay']
epsilon_min = cfg.learning_hyperparameters['epsilon_min']
epsilon_history = []

# Data collection lists
observations_H = []
actions_H = []
agents_route_H = []
loss_H = []
step_distances = []

progress_bar = tqdm.trange(cfg.episode_num, desc='Progress', leave=True)

for ep in progress_bar:

    done = False
    terminate = False
    observation = env.reset()
    # print(f"observation shape: {observation.shape}")
    rewards_this_episode = []

    while not done and not terminate:
        old_observation = copy(observation)

        # epsilon-greedy algorithm
        if (np.random.random() < epsilon) or (cfg.algorithm == 'random'):
            actions = np.array([env.action_space.sample()
                                for _ in range(env.N)])  # Random actions
        else:
            actions = np.array([d3ql_algorithm.get_model_output(observation[i, :, :].reshape(1, *dimension), i)
                                for i in range(env.N)])  # Intelligent actions

        observation, rewards, terminate, info, done = env.step(actions)

        rewards_this_episode.append(rewards.mean())

        actions_encoded = np.array(
            [utils.return_one_hot_vector(a) for a in actions])
        buffer.store_experience(old_observation,
                                observation,
                                actions_encoded, rewards, done or terminate)

        # Collecting observation, action for CSV logging
        for i in range(cfg.N):
            observations_H.append({'No.': len(observations_H), 'episode_num': ep, 'agent ID': i + 1,
                                   'observation': old_observation[i].tolist()})
            actions_H.append({'No.': len(actions_H), 'episode_num': ep, 'agent ID': i + 1, 'action': actions[i]})

        # update epsilon
        epsilon *= epsilon_decay
        epsilon = max(epsilon, epsilon_min)
        epsilon_history.append(epsilon)

        # train models independently
        if buffer.mem_counter > cfg.batch_size:
            state, next_state, action, reward, dones = buffer.sample_buffer()

            if cfg.algorithm != 'random':
                loss = d3ql_algorithm.train_independent(
                    state, next_state, action, reward, dones)
                loss_H.append(loss)

        # Record distance for each step
        step_distances.append(info['distances'])

    # Collecting route for CSV logging  
    for i in range(cfg.T):
        row = {'No.': len(agents_route_H) + 1, 'step_num': i, 'episode_num': ep}
        for agent_id in info['agents_route'].keys():
            row[f'agent ID {agent_id}'] = info['agents_route'][agent_id][i] if i < len(info['agents_route'][agent_id]) else None
        agents_route_H.append(row)

    is_successful[ep] = done
    average_rewards[ep] = np.array(rewards_this_episode).mean() if len(
        rewards_this_episode) > 0 else 0
    distance_all_value = [
        value for sublist in info['distances'].values() for value in sublist]
    distance_episodes[ep] = sum(distance_all_value) / len(distance_all_value)
    rate_distance_over_step[ep] = distance_episodes[ep] / info['steps_number']

    progress_bar.set_description(f"Avg. Distance: {int(distance_episodes[ep])}")
    progress_bar.refresh()

print('*****************************************')
print(f'Success rate for {cfg.episode_num} episodes was {is_successful.mean() * 100}%')
print(f'Average reward for {cfg.episode_num} episodes was {round(average_rewards.mean(), 3)}')
print('*****************************************')

d3ql_algorithm.save_models()

output_folder = 'output_results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print_file_path = os.path.join(output_folder, 'results.txt')
with open(print_file_path, 'w') as f:
    f.write(f'Success rate for {cfg.episode_num} episodes was {is_successful.mean() * 100}%\n')
    f.write(f'Average reward for {cfg.episode_num} episodes was {round(average_rewards.mean(), 3)}')

plt.figure('Reward')
if len(average_rewards) < cfg.episode_num:
    average_rewards = np.pad(average_rewards, (0, cfg.episode_num -
                                               len(average_rewards)), 'constant', constant_values=np.nan)
plt.plot(range(1, cfg.episode_num + 1), average_rewards, marker='.')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Average Rewards Over Episodes')
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'Rewards_over_Episodes.png'))

plt.figure('Distance')
if len(average_rewards) < cfg.episode_num:
    average_rewards = np.pad(average_rewards, (0, cfg.episode_num -
                                               len(average_rewards)), 'constant', constant_values=np.nan)
plt.plot(range(1, cfg.episode_num + 1), distance_episodes, marker='.')
plt.xlabel('Episode')
plt.ylabel('Distance')
plt.title('Average Distances Over Episodes')
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'Distances_over_Episodes.png'))

plt.figure('Epsilon')
plt.plot(range(1, len(epsilon_history) + 1), epsilon_history, marker='.')
plt.xlabel('Update Times')
plt.ylabel('Epsilon')
plt.title('Epsilon Trend')
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'Epsilon_Trend.png'))

plt.figure('Rate')
if len(rate_distance_over_step) < cfg.episode_num:
    rate_distance_over_step = np.pad(rate_distance_over_step, (0, cfg.episode_num - len(
        rate_distance_over_step)), 'constant', constant_values=np.nan)
plt.plot(range(1, cfg.episode_num + 1), rate_distance_over_step, marker='.')
plt.xlabel('Episode')
plt.ylabel('distance/step_num')
plt.title('Rate: Distance / Step_num')
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'Rate.png'))

plt.figure('Loss')
plt.plot(range(1, len(loss_H) + 1), loss_H, marker='.')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Loss Over Training Steps')
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'Loss.png'))

# Save step distances as .npy file
np.save(os.path.join(output_folder, 'distances.npy'), step_distances)


def save_to_csv(filename, data, fieldnames):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


observation_fieldnames = ['No.', 'episode_num', 'agent ID', 'observation']
action_fieldnames = ['No.', 'episode_num', 'agent ID', 'action']
route_fieldnames = ['No.', 'step_num', 'episode_num'] + [f'agent ID {i}' for i in info['agents_route'].keys()]

save_to_csv(os.path.join(output_folder, 'observations.csv'), observations_H, observation_fieldnames)
save_to_csv(os.path.join(output_folder, 'actions.csv'), actions_H, action_fieldnames)
save_to_csv(os.path.join(output_folder, 'agents_route.csv'), agents_route_H, route_fieldnames)
