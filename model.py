from abc import ABC

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim

import config as cfg

th.cuda.empty_cache()


# noinspection PyUnresolvedReferences
class ReplayBuffer(ABC):

    def __init__(self, capacity, num_agents, dimension):
        super().__init__()

        self.capacity = capacity
        self.num_agents = num_agents
        self.dimension = dimension

        self.num_actions = cfg.learning_hyperparameters['num_actions']

        self.device = 'cuda' if th.cuda.is_available() else 'cpu'

        self.state_memory = np.zeros(
            (self.capacity, self.num_agents, self.dimension * self.dimension),
            dtype=np.float32)
        self.next_state_memory = np.zeros(
            (self.capacity, self.num_agents, self.dimension * self.dimension),
            dtype=np.float32)
        self.action_memory = np.zeros((self.capacity, self.num_agents, self.num_actions), dtype=np.int64)
        self.reward_memory = np.zeros((self.capacity, self.num_agents), dtype=np.float32)
        self.terminal_memory = np.zeros((self.capacity, self.num_agents), dtype=bool)

        self.mem_counter = 0

    def store_experience(self, state, next_state, action, reward, done):
        index = self.mem_counter % self.capacity

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def sample_buffer(self):
        max_mem = min(self.mem_counter, self.capacity)

        # "replace=False" assures that no repetitive memory is selected in batch
        batch = np.random.choice(max_mem, cfg.batch_size, replace=False)

        states = th.tensor(self.state_memory[batch]).to(self.device)
        next_state = th.tensor(self.next_state_memory[batch]).to(self.device)
        actions = th.tensor(self.action_memory[batch]).to(self.device)
        rewards = th.tensor(self.reward_memory[batch]).to(self.device)
        terminal = th.tensor(self.terminal_memory[batch]).to(self.device)

        experience = [states, next_state, actions, rewards, terminal]

        return experience

    def flush(self, config):
        self.__init__(config)

    def get_all_data(self):
        return self.state_memory, self.next_state_memory, \
            self.action_memory, self.reward_memory, self.terminal_memory, \
            self.mem_counter

    def set_all_data(self, state_memory, next_state_memory,
                     action_memory, reward_memory, terminal_memory, mem_counter):
        self.state_memory = state_memory
        self.next_state_memory = next_state_memory
        self.action_memory = action_memory
        self.reward_memory = reward_memory
        self.terminal_memory = terminal_memory
        self.mem_counter = mem_counter


# noinspection PyUnresolvedReferences
class DeepQNetwork(nn.Module):
    # Reference: https://github.com/mshokrnezhad/Dueling_for_DRL

    def __init__(self, num_agents, dimension):
        nn.Module.__init__(self)

        self.num_agents = num_agents
        self.dimension = dimension
        self.fc_sizes = cfg.learning_hyperparameters['fc_sizes']
        self.learning_rate = cfg.learning_hyperparameters['learning_rate']
        self.num_actions = cfg.learning_hyperparameters['num_actions']

        self.device = 'cuda' if th.cuda.is_available() else 'cpu'

        # Build the Modules
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(self.dimension ** 2, self.fc_sizes[0])
        self.fc_2 = nn.Linear(self.fc_sizes[0], self.fc_sizes[1])
        self.fc_3 = nn.Linear(self.fc_sizes[1], self.fc_sizes[2])

        self.V = nn.Linear(self.fc_sizes[2], 1)
        self.A = nn.Linear(self.fc_sizes[2], self.num_actions)

        # self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate,
                                    amsgrad=True, weight_decay=0.001)
        self.loss = nn.MSELoss()

        self.to(self.device)  # move whole model to the device

    def forward(self, state):
        # forward propagation includes defining layers

        # features, _ = self.lstm(state)
        # x = self.relu(features[:, -1, :])

        x = self.relu(self.fc_1(state))
        x = self.relu(self.fc_2(x))
        x = self.relu(self.fc_3(x))

        V = self.V(x)
        A = self.A(x)

        return V, A

    def save_checkpoint(self, checkpoint_file):
        th.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(th.load(checkpoint_file))


# noinspection PyUnresolvedReferences
class D3QL:
    # Reference: https://github.com/mshokrnezhad/Dueling_for_DRL

    def __init__(self, num_agents, dimension):

        self.num_agents = num_agents
        self.dimension = dimension

        self.device = 'cuda' if th.cuda.is_available() else 'cpu'

        self.gamma = cfg.learning_hyperparameters['gamma']
        self.replace_target_interval = cfg.learning_hyperparameters['replace_target_interval']

        self.loss = nn.MSELoss()

        # create one model per agent
        self.models = np.empty(self.num_agents, dtype=object)
        self.target_models = np.empty(self.num_agents, dtype=object)
        self.models_initial_weights = np.empty(self.num_agents, dtype=object)

        for i in range(self.num_agents):
            self.models[i] = DeepQNetwork(self.num_agents, self.dimension)
            self.target_models[i] = DeepQNetwork(self.num_agents, self.dimension)

            if cfg.load_pretrained_model:
                file = f'results/{self.folder_name}/algo_{self.algorithm}/model_{i}.pt'
                self.models[i].load_checkpoint(file)

            # copy initial weights to target models
            self.models_initial_weights[i] = self.models[i].state_dict()
            self.target_models[i].load_state_dict(self.models_initial_weights[i])

        # used for updating target networks
        self.learn_step_counter = 0

        self.indexes = np.arange(cfg.batch_size)

    def __replace_target_networks(self, model_index):
        if self.learn_step_counter == 0 \
                or (self.learn_step_counter % self.replace_target_interval) == 0:
            self.target_models[model_index].load_state_dict(self.models[model_index].state_dict())

    @staticmethod
    def __convert_value_advantage_to_q_values(v, a):
        return th.add(v, (a - a.mean(dim=1, keepdim=True)))

    def get_model_output(self, observation, i):

        observation = th.tensor(observation, dtype=th.float).to(self.device).unsqueeze(0)
        value, advantages = self.models[i].forward(observation)

        action = th.argmax(advantages).item()

        return action

    def train_independent(self, states, next_states, actions, reward, dones):

        q_predicted = th.zeros((cfg.batch_size, self.num_agents)).to(self.device)
        q_next = th.zeros((cfg.batch_size, self.num_agents)).to(self.device)

        for i in range(self.num_agents):
            # initialize local models
            self.models[i].train()
            self.models[i].optimizer.zero_grad()
            self.__replace_target_networks(model_index=i)

            V_states, A_states = self.models[i].forward(states[:, i, :])
            actions_num = np.nonzero(actions[:,i,:])[:, 1:]
            q_values = self.__convert_value_advantage_to_q_values(V_states, A_states)
            for d in self.indexes:
                # Ensure the correct indexing
                action_index = actions_num[d, 0]  # get the specific action index
                q_predicted[d, i] = q_values[d, action_index]  # use the specific indices for indexing

            _, A_next_states = self.models[i].forward(next_states[:, i, :])
            actions_states_best = A_next_states.argmax(axis=1).detach()

            V_next_states, A_next_states = self.target_models[i].forward(next_states[:, i, :])
            q_next_all_actions = self.__convert_value_advantage_to_q_values(V_next_states, A_next_states)
            q_next[:, i] = q_next_all_actions.gather(1, actions_states_best.unsqueeze(1)).squeeze()
            q_next[dones[:, i], i] = 0.0

        total_target = th.nan_to_num(reward).mean(axis=-1).unsqueeze(-1) + (self.gamma * q_next)

        loss = self.loss(q_predicted, total_target).to(self.device)
        loss.backward()

        for i in range(self.num_agents):
            self.models[i].optimizer.step()
            self.models[i].eval()

        self.learn_step_counter += 1

        return loss.detach().cpu().numpy()

    def get_weights(self):
        return self.model.state_dict(), self.target_model.state_dict()

    def set_weights(self, weights, weights_target):
        self.model.load_state_dict(weights)
        self.target_model.load_state_dict(weights_target)

        self.model.lstm.flatten_parameters()
        self.target_model.lstm.flatten_parameters()

    def reset_models(self):
        ...
