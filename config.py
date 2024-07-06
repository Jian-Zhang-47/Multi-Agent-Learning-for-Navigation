# Environment

L = 5  # Grid size
P = 0.1  # Number of obstacles (as a percentage to grid size)
N = 2  # Number of agents
T = 50  # Maximum episode length
M = 3  # Agent FoV size

# RL
episode_num = 1000
memory_size = 1000
batch_size = 32

learning_hyperparameters = {
    'fc_sizes': [32, 16, 8],
    'learning_rate': 0.001,
    'num_actions': 4,
    'gamma': 0.9,
    'replace_target_interval': 50,

}

load_pretrained_model = False

replacement_dict = {'  ': 0, 'S-': -1, 'S+': -2, 'S0': -3}
