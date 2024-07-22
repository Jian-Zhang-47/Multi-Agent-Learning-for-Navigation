import csv
import os
from copy import copy

import numpy as np
import tqdm
from matplotlib import pyplot as plt

import utils
from GridNavigationEnv import GridNavigationEnv
from model import D3QL
from model import ReplayBuffer


def save_to_csv(filename, data, fieldnames):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


# noinspection PyUnresolvedReferences
class SimulationRunner:

    def __init__(self, cfg):

        self.cfg = cfg

        self.env = GridNavigationEnv()
        self.observation = self.env.reset()

        self.fov_size = (2 * self.cfg.M) + 1
        self.dimension = (3, self.fov_size, self.fov_size)
        self.buffer = ReplayBuffer(self.cfg.memory_size, self.cfg.N, self.dimension)
        self.d3ql_algorithm = D3QL(self.cfg.N, self.dimension)

        self.is_successful = np.zeros(self.cfg.episode_num)
        self.average_rewards = np.zeros(self.cfg.episode_num)
        self.distance_episodes = np.zeros(self.cfg.episode_num)
        self.rate_distance_over_step = np.zeros(self.cfg.episode_num)

        self.epsilon = self.cfg.learning_hyperparameters['epsilon']
        self.epsilon_decay = self.cfg.learning_hyperparameters['epsilon_decay']
        self.epsilon_min = self.cfg.learning_hyperparameters['epsilon_min']
        self.epsilon_history = []

        # Data collection lists
        self.observations_H = []
        self.actions_H = []
        self.agents_route_H = []
        self.loss_H = []
        self.step_distances = []

        self.progress_bar = tqdm.trange(self.cfg.episode_num, desc='Progress', leave=True)

    def run_one_episode(self):

        for ep in self.progress_bar:

            done = False
            terminate = False
            observation = self.env.reset()
            # print(f"observation shape: {observation.shape}")
            rewards_this_episode = []

            while not done and not terminate:
                old_observation = copy(observation)

                # epsilon-greedy algorithm
                if (np.random.random() < self.epsilon) or (self.cfg.algorithm == 'random'):
                    actions = np.array([self.env.action_space.sample()
                                        for _ in range(self.env.N)])  # Random actions
                else:
                    actions = np.array(
                        [self.d3ql_algorithm.get_model_output(observation[i, :, :].reshape(1, *self.dimension), i)
                         for i in range(self.env.N)])  # Intelligent actions

                observation, rewards, terminate, info, done = self.env.step(actions)

                rewards_this_episode.append(rewards.mean())

                actions_encoded = np.array(
                    [utils.return_one_hot_vector(a) for a in actions])
                self.buffer.store_experience(old_observation.reshape((self.cfg.N, *self.dimension)),
                                             observation.reshape((self.cfg.N, *self.dimension)),
                                             actions_encoded, rewards, done or terminate)

                # Collecting observation, action for CSV logging
                for i in range(self.cfg.N):
                    self.observations_H.append({'No.': len(self.observations_H),
                                                'episode_num': ep,
                                                'agent ID': i + 1,
                                                'observation': old_observation[i].tolist()})
                    self.actions_H.append(
                        {'No.': len(self.actions_H),
                         'episode_num': ep,
                         'agent ID': i + 1,
                         'action': actions[i]})

                # update epsilon
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)
                self.epsilon_history.append(self.epsilon)

                # train models independently
                if self.buffer.mem_counter > self.cfg.batch_size:
                    state, next_state, action, reward, dones = self.buffer.sample_buffer()

                    if self.cfg.algorithm != 'random':
                        loss = self.d3ql_algorithm.train_independent(
                            state, next_state, action, reward, dones)
                        self.loss_H.append(loss)

                # Record distance for each step
                self.step_distances.append(info['distances'])

            # Collecting route for CSV logging
            for i in range(self.cfg.T):
                row = {'No.': len(self.agents_route_H) + 1, 'step_num': i, 'episode_num': ep}
                for agent_id in info['agents_route'].keys():
                    row[f'agent ID {agent_id}'] = info['agents_route'][agent_id][i] if i < len(
                        info['agents_route'][agent_id]) else None
                self.agents_route_H.append(row)

            self.is_successful[ep] = done
            self.average_rewards[ep] = np.array(rewards_this_episode).mean() if len(
                rewards_this_episode) > 0 else 0
            distance_all_value = [
                value for sublist in info['distances'].values() for value in sublist]
            self.distance_episodes[ep] = sum(distance_all_value) / len(distance_all_value)
            self.rate_distance_over_step[ep] = self.distance_episodes[ep] / info['steps_number']

            self.progress_bar.set_description(f"Avg. Distance: {int(self.distance_episodes[ep])}")
            self.progress_bar.refresh()

    def save_results(self, name=None):
        print('*****************************************')
        print(f'Success rate for {self.cfg.episode_num} episodes was {self.is_successful.mean() * 100}%')
        print(f'Average reward for {self.cfg.episode_num} episodes was {round(self.average_rewards.mean(), 3)}')
        print('*****************************************')

        output_folder = f'output_results_{name}' if name is not None else 'output_results'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.d3ql_algorithm.save_models()
        print_file_path = os.path.join(output_folder, 'results.txt')
        with open(print_file_path, 'w') as f:
            f.write(f'Success rate for {self.cfg.episode_num} episodes was {self.is_successful.mean() * 100}%\n')
            f.write(f'Average reward for {self.cfg.episode_num} episodes was {round(self.average_rewards.mean(), 3)}')

        plt.figure('Reward')
        if len(self.average_rewards) < self.cfg.episode_num:
            average_rewards = np.pad(self.average_rewards, (0, self.cfg.episode_num -
                                                            len(self.average_rewards)), 'constant',
                                     constant_values=np.nan)
        plt.plot(range(1, self.cfg.episode_num + 1), self.average_rewards, marker='.')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Average Rewards Over Episodes')
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, 'Rewards_over_Episodes.png'))

        plt.figure('Distance')
        if len(self.average_rewards) < self.cfg.episode_num:
            average_rewards = np.pad(self.average_rewards, (0, self.cfg.episode_num -
                                                            len(self.average_rewards)), 'constant',
                                     constant_values=np.nan)
        plt.plot(range(1, self.cfg.episode_num + 1), self.distance_episodes, marker='.')
        plt.xlabel('Episode')
        plt.ylabel('Distance')
        plt.title('Average Distances Over Episodes')
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, 'Distances_over_Episodes.png'))

        plt.figure('Epsilon')
        plt.plot(range(1, len(self.epsilon_history) + 1), self.epsilon_history, marker='.')
        plt.xlabel('Update Times')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Trend')
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, 'Epsilon_Trend.png'))

        plt.figure('Rate')
        if len(self.rate_distance_over_step) < self.cfg.episode_num:
            rate_distance_over_step = np.pad(self.rate_distance_over_step, (0, self.cfg.episode_num - len(
                self.rate_distance_over_step)), 'constant', constant_values=np.nan)
        plt.plot(range(1, self.cfg.episode_num + 1), self.rate_distance_over_step, marker='.')
        plt.xlabel('Episode')
        plt.ylabel('distance/step_num')
        plt.title('Rate: Distance / Step_num')
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, 'Rate.png'))

        plt.figure('Loss')
        plt.plot(range(1, len(self.loss_H) + 1), self.loss_H, marker='.')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Loss Over Training Steps')
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, 'Loss.png'))

        # Save step distances as .npy file
        np.save(os.path.join(output_folder, 'distances.npy'), self.step_distances)

        # todo: need to be fixed
        # observation_fieldnames = ['No.', 'episode_num', 'agent ID', 'observation']
        # action_fieldnames = ['No.', 'episode_num', 'agent ID', 'action']
        # route_fieldnames = (['No.', 'step_num', 'episode_num']
        #                     + [f'agent ID {i}' for i in self.agents_route_H[-1].keys()])
        #
        # save_to_csv(os.path.join(output_folder, 'observations.csv'), self.observations_H, observation_fieldnames)
        # save_to_csv(os.path.join(output_folder, 'actions.csv'), self.actions_H, action_fieldnames)
        # save_to_csv(os.path.join(output_folder, 'agents_route.csv'), self.agents_route_H, route_fieldnames)
