from gymnasium import spaces

# simple, variable_M, ...
scenario = 'simple'

# Environment

# 'random', 'independent_D3QL', 'CTDE_D3QL'
algorithm = 'random'

L = 100  # Grid size
P = 0.1  # Number of obstacles (as a ratio of grid size)
N = 1  # Number of agents
T = 1000  # Maximum episode length
M = 50  # Agent FoV size
view_angle = 360
move_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# RL
episode_num = 1000
memory_size = 100
batch_size = 32
new_grid_per_episode = False

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
