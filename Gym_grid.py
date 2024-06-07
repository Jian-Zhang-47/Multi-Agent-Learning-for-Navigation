import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class GridNavigationEnv(gym.Env):
    def __init__(self, L=6, P=0.3, N=1, T=100):
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
        self.rewards = [0]*self.N # Rewards of agents

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # Four possible actions: up, down, left, right
        #self.observation_space = spaces.Box(low=0, high=max(N, 1), shape=(L, L), dtype=np.int32)
        
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
        if 0 <= nx < self.L and 0 <= ny < self.L and self.grid[nx, ny] == 0:
            self.grid[x, y] = 0
            self.grid[nx, ny] = agent_id
            self.agents[agent_id - 1] = (nx, ny)
            self.agents_route_dict[agent_id].append(self.agents[agent_id - 1])
        else:
            self.agents_route_dict[agent_id].append((x, y))

    def reset(self):
        self.grid = np.zeros((self.L, self.L), dtype=int)
        self.agents = []
        self.agents_id_list = []
        self.agents_route_dict = {}
        self.destination = None
        self.steps = 0
        self.init_environment()
        return self.grid

    def step(self, actions):
        for agent_id, action in enumerate(actions, start=1):
            self.move_agent(agent_id, action)
            if self.agents[agent_id - 1] == self.destination:
                self.rewards[agent_id-1] += self.T # Reward for reaching destination
            else:
                self.rewards[agent_id-1] += -1 # Penalty for each move
        self.steps += 1
        done = all(self.agents[agent_id - 1] == self.destination for agent_id in self.agents_id_list)
        terminate = self.steps > self.T
        return self.grid, self.rewards, done, terminate, self.agents_route_dict

    def render(self, mode='human'):
        for row in self.grid:
            print('   '.join(str(x) for x in row))



if __name__ == "__main__":
    env = GridNavigationEnv(L=3, P=0.3, N=1, T=10)
    obs = env.reset()
    done = False
    terminate = False

while not done and not terminate:
    actions = [env.action_space.sample() for _ in range(env.N)]  # Random actions
    obs, reward, done, terminate, info = env.step(actions)
    env.render()
    print(f"Reward: {reward}")
    print(f"Route: {info}\n")
