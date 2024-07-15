from gymnasium import spaces
# Environment

L = 100  # Grid size
P = 0.1  # Number of obstacles (as a percentage to grid size)
N = 1  # Number of agents
T = 1000  # Maximum episode length
M = 50  # Agent FoV size
action_space = spaces.Discrete(4)

# RL
episode_num = 1000
memory_size = 100
batch_size = 32

learning_hyperparameters = {
    'fc_sizes': [32, 16, 8],
    'learning_rate': 0.001,
    'num_actions': 4,
    'gamma': 0.9,
    'replace_target_interval': 50,
    'epsilon': 1,
    'epsilon_decay': 0.99,
    'epsilon_min': 0.05

}

load_pretrained_model = False

replacement_dict = {'  ': 0, 'S-': -1, 'S+': -2, 'S0': -3}

checkpoint_dir = 'checkpoint_folder'
output_dir = 'output_folder'
