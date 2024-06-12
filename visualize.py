import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from copy import copy

from Gym_grid import GridNavigationEnv


def save_frames_as_gif(frames, file):
    plt.figure(figsize=(frames[0].shape[1] / 10, frames[0].shape[0] / 10), dpi=120)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(file, writer='imagemagick', fps=60)



# Parameters
L = 8  # Grid size
P = 0.1 # Percentage of obstacles
N = 2  # Number of agents
T = 100  # Maximum episode length
M = 2  # Agent FoV size

# Create environment and run simulation
env = GridNavigationEnv(L, P, N, T, M)
grid_map = env.reset()
done = False
terminate = False
#background = copy(grid_map) # GIF shows agents' positions in step 0

while not done and not terminate:
    actions = [env.action_space.sample() for _ in range(env.N)]  # Random actions
    grid_map, rewards, done, terminate, routes, steps, destination= env.step(actions)
print(f"Destination: {destination}")
print(f"Reward: {rewards}")
print(f"All agents have reached the destination: {done}")
print(f"Some agents are stuck somewhere: {terminate}")
print(f"Route: {routes}\n")
env.grid[env.destination[0],env.destination[1]] = N+1 # Add destination's position in grip
background = copy(grid_map) # GIF shows agents' positions in the last step
frames = []


for t in range(env.T):

    grid_with_nodes = copy(background)

    for i in range(env.N):

        if t < len(routes[i+1]):
            node_loc = routes[i+1][t]

        grid_with_nodes[node_loc[0], node_loc[1]] = i+1
        
    frames.append(grid_with_nodes)


save_frames_as_gif(frames, 'anim.gif')