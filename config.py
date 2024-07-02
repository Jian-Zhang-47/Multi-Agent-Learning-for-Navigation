# Environment

L = 20  # Grid size
P = 0.2  # Number of obstacles (as a percentage to grid size)
N = 10  # Number of agents
T = 10  # Maximum episode length
M = 10  # Agent FoV size

# RL
episode_num = 10
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

replacement_dict = {'  ': 0, 'S-': 1, 'S+': -1, 'S0': 2}
