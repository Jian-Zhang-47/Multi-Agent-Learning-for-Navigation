import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class GridNavigationEnv(gym.Env):
    def __init__(self, L=6, P=0.3, N=1, T=100, M = 6):
        super(GridNavigationEnv, self).__init__()
        self.L = L  # Grid size
        self.P = P  # Number of obstacles as a percentage of grid size
        self.N = N  # Number of agents
        self.T = T  # Maximum episode length
        #self.M = M  # Agent FoV size
        self.grid = np.zeros((L, L), dtype=int)
        self.obstacles = int(P * L * L)
        self.agents = []    # Coordinates of agents
        self.agents_id_list = []    # List of agent IDs
        self.agents_route_dict = {} # Route coordinates of agents
        self.destination = None
        self.steps = 0  # Number of steps
        self.rewards_dict = {i+1: [] for i in range(self.N)} # Rewards of agents

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # Four possible actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=max(N, 1), shape=(L, L), dtype=np.int32)
        
        self.init_environment()

    def init_environment(self):
        self.place_obstacles()
        self.place_destination()
        self.place_agents()

    def place_obstacles(self):
        for _ in range(self.obstacles):
            while True:
                x, y = random.randint(0, self.L - 1), random.randint(0, self.L - 1)
                if self.grid[x, y] == 0:
                    self.grid[x, y] = -1  # Obstacle grid value is -1
                    break

    def place_destination(self):
        while True:
            x, y = random.randint(0, self.L - 1), random.randint(0, self.L - 1)
            if self.grid[x, y] == 0:
                self.destination = (x, y)  # Destination grid value is 0
                break

    def place_agents(self):
        for i in range(self.N):
            while True:
                x, y = random.randint(0, self.L - 1), random.randint(0, self.L - 1)
                if self.grid[x, y] == 0:
                    self.agents_id_list.append(i + 1)  # Agent ID
                    self.grid[x, y] = self.agents_id_list[i]
                    self.agents.append((x, y))
                    self.agents_route_dict[self.agents_id_list[i]] = [self.agents[i]]  # Record the original position of agents
                    break

    def move_agent(self, agent_id, action):
        x, y = self.agents[agent_id - 1]
        if action == 0:  # Up
            dx, dy = -1, 0
        elif action == 1:  # Down
            dx, dy = 1, 0
        elif action == 2:  # Left
            dx, dy = 0, -1
        elif action == 3:  # Right
            dx, dy = 0, 1

        nx, ny = x + dx, y + dy
        self.grid[self.destination] = 0
        if 0 <= nx < self.L and 0 <= ny < self.L and self.grid[nx, ny] == 0:
            self.grid[x, y] = 0
            self.grid[nx, ny] = agent_id
            self.agents[agent_id - 1] = (nx, ny)
            self.agents_route_dict[agent_id].append(self.agents[agent_id - 1])  # Record the new position of agents
        else:
            self.agents_route_dict[agent_id].append((x, y))

    def reset(self):
        self.grid = np.zeros((self.L, self.L), dtype=int)
        self.agents = []
        self.agents_id_list = []
        self.agents_route_dict = {}
        self.destination = None
        self.steps = 0
        self.rewards_dict = {i+1: [] for i in range(self.N)}
        self.init_environment()
        return self.grid

    def step(self, actions):
        for a_id in self.agents_id_list:
            if self.agents[a_id-1] == self.destination:
                self.agents_id_list.remove(a_id)
            else:
                for agent,action in enumerate(actions, start=1):
                    if a_id == agent:
                        self.move_agent(a_id, action)
                        x, y = self.agents[a_id - 1]
                        dest_x, dest_y = self.destination
                        distance = abs(x - dest_x) + abs(y - dest_y)
                        if distance == 0:
                            self.rewards_dict[a_id].append(self.T) # Reward for reaching destination
                            self.agents_id_list.remove(a_id)
                        else:
                            self.rewards_dict[a_id].append(round(1/(distance + 1),4)) # Penalty for each move
        self.steps += 1
        done = all(self.agents[agent_id - 1] == self.destination for agent_id in self.agents_id_list)
        terminate = self.steps >= self.T
        return self.grid, self.rewards_dict, done, terminate, self.agents_route_dict, self.steps, self.destination

    def render(self, mode='human'):
        for row in self.grid:
            print('   '.join(str(x) for x in row))


if __name__ == "__main__":
    env = GridNavigationEnv(L=4, P=0.1, N=3, T=10, M=4)
    grid_map = env.reset()
    done = False
    terminate = False
    print(f'In 0 step')
    for row in env.grid:
        print('   '.join(str(x) for x in row))
    print()

while not done and not terminate:
    actions = [env.action_space.sample() for _ in range(env.N)]  # Random actions
    grid_map, rewards, done, terminate, routes, steps, destination= env.step(actions)
    print(f'In {steps} steps')
    env.render()
    print('')
print(f"Destination: {destination}")
print(f"Reward: {rewards}")
print(f"All agents have reached the destination: {done}")
print(f"Some agents are stuck somewhere: {terminate}")
print(f"Route: {routes}\n")
