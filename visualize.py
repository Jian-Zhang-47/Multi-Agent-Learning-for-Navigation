import matplotlib.pyplot as plt
from matplotlib import animation
from copy import copy
from FoV_angle import GridNavigationEnv
import matplotlib.pyplot as plt


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
    observation, rewards, terminate, info, done= env.step(actions)
    print(observation)
    print(f'In {info['steps_number']} steps')
    env.render()
    env.render_fov(observation)
    print('')
print(f"Destination: {info['destination']}")
print(f"Reward: {rewards}")
print(f"All agents have reached the destination: {done}")
print(f"Some agents are stuck somewhere: {terminate}")
print(f"Route: {info['agents_route']}\n")
env.grid[env.destination[0],env.destination[1]] = N+1 # Add destination's position in grip
background = copy(info['grid_map']) # GIF shows agents' positions in the last step
frames = []

# Save as a gif to show the routes
def save_frames_as_gif(frames, file):
    plt.figure(figsize=(frames[0].shape[1] / 10, frames[0].shape[0] / 10), dpi=120)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(file, writer='pillow', fps=60)

for t in range(env.T):
    grid_with_nodes = copy(background)
    for i in range(env.N):
        if t < len(info['agents_route'][i+1]):
            node_loc = info['agents_route'][i+1][t]
        grid_with_nodes[node_loc[0], node_loc[1]] = i+1 
    frames.append(grid_with_nodes)
save_frames_as_gif(frames, 'Routes.gif')


# Graph to show the distance between agent and destination
fig1, ax1 = plt.subplots()
x = list(range(T))
for i in range(N):
    y = info['distances'][i+1]
    if len(y)<len(x):
        y.extend([None] * (len(x) - len(y)))
    ax1.plot(x,y,label=f'Agent {i+1}')
ax1.set_xlabel('t')
ax1.set_ylabel('d')
ax1.set_title('Distance between agent and destination')
ax1.legend()
plt.savefig('Distance.png')

# Graph to show the numbers of step
fig2, ax2 = plt.subplots()
x = []
y = []
for i in range(N):
    x.append(f'{i+1}')
    y.append(len(info['agents_route'][i+1]))
ax2.bar(x,y)
ax2.set_xlabel('agent ID')
ax2.set_ylabel('Numbers of step')
ax2.set_title('Numbers of agents\' step')
plt.savefig('Step.png')