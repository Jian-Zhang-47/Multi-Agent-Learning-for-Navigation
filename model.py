from abc import ABC

import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
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
            (self.capacity,self.num_agents, *self.dimension),
            dtype=np.float32)
        self.next_state_memory = np.zeros(
            (self.capacity, self.num_agents, *self.dimension),
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
    def __init__(self, name, input_dims):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = cfg.checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.learning_rate = cfg.learning_hyperparameters['learning_rate']
        self.num_actions = cfg.learning_hyperparameters['num_actions']

        # convolutions to process observations and pass then to fully connected layers
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)  # input_dims[0]: number of channels, 32: number of 
        # outgoing filters, 8: kernel size (8*8 pixels)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)  # 32: number of incoming filters, 64: number of outgoing filters
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)  # convolutions to process observations and pass then to fully connected layers

        processed_input_dims = self.calculate_output_dims(input_dims)
        
        self.fc1 = nn.Linear(processed_input_dims, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, self.num_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')  # use GPU if available
        self.to(self.device)  # move whole model to device

    def calculate_output_dims(self, input_dims):
        state = th.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)

        return int(np.prod(dims.size()))  # np.prod: to return the product of array elements over a given axis.

    def forward(self, state):  # forward propagation includes defining layers
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))  # conv3 shape is batch size * number of filters * H * W (of output image)

        conv_state = conv3.view(conv3.size()[0], -1) # means that get the first dim and flatten others

        flat1 = F.relu(self.fc1(conv_state))
        V = self.V(flat1)
        A = self.A(flat1)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        th.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(th.load(self.checkpoint_file))


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
            # lr, n_actions, name, input_dims, checkpoint_dir
            self.models[i] = DeepQNetwork(f'model_{i}', self.dimension)
            self.target_models[i] = DeepQNetwork(f'target_model_{i}', self.dimension)

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

        observation = th.tensor(observation, dtype=th.float).to(self.device).view(1, *self.dimension)
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
    
    def save_models(self):
        for i in range(self.num_agents):
            file = f'output_results/model_{i}.pt'
            self.models[i].save_checkpoint(file)
            
    def get_weights(self):
        return self.model.state_dict(), self.target_model.state_dict()

    def set_weights(self, weights, weights_target):
        self.model.load_state_dict(weights)
        self.target_model.load_state_dict(weights_target)

        self.model.lstm.flatten_parameters()
        self.target_model.lstm.flatten_parameters()

    def reset_models(self):
        ...
