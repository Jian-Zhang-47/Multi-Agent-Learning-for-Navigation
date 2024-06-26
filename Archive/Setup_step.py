import random
import numpy as np

class GridEnvironment:
    def __init__(self, L, P, N, T, M):
        self.L = L  # Grid size
        self.P = P  # Number of obstacles as a percentage of grid size
        self.N = N  # Number of agents
        self.T = T  # Maximum episode length
        self.M = M  # Agent FoV size
        self.grid = np.zeros((L, L), dtype=int)
        self.obstacles = int(P * L * L)
        self.agents = []
        self.agents_id_list = []
        self.destination = None
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
                    self.grid[x, y] = -1  # Obstacle = -1
                    ##print(f"Obstacle are {x,y}")
                    break

    def place_destination(self):
        while True:
            x, y = random.randint(0, self.L - 1), random.randint(0, self.L - 1)
            if self.grid[x, y] == 0:
                self.destination = (x, y)  # Destination = 0
                print(f"Destination is {x,y}")
                break

    def place_agents(self):
        for i in range(self.N):
            while True:
                x, y = random.randint(0, self.L - 1), random.randint(0, self.L - 1)
                if self.grid[x, y] == 0:
                    self.agents_id_list.append(i+1)   # Agent ID
                    self.grid[x, y] =  self.agents_id_list[i]
                    self.agents.append((x, y))
                    
                    #print(f"agent {i+1} is {x,y}")
                    break

    def move_agent(self, agent_id):
        x, y = self.agents[agent_id-1]
        possible_moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(possible_moves)
        for dx, dy in possible_moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.L and 0 <= ny < self.L and self.grid[nx, ny] == 0:
                self.grid[x, y] = 0
                self.grid[nx, ny] = agent_id
                self.agents[agent_id-1] = (nx, ny)
                print(f"Agent {agent_id} moves step {dx,dy} from {x,y} into {nx,ny}")
                break

    def run_simulation(self):
        rest_agents_id_list = self.agents_id_list
        x = 0
        for t in range(self.T):
            t_agents_id_list= rest_agents_id_list
            print(t_agents_id_list)
            for a_id in t_agents_id_list:
                if self.agents[a_id-1] == self.destination:
                    print(f"Agent {a_id} reached the destination in 0 steps.")
                    self.grid[self.agents[a_id-1]] = 0
                    rest_agents_id_list.remove(a_id)
                    x+=1

                else:
                    print(f"Time = {t+1}=============================")
                    self.move_agent(a_id)
                    if self.agents[a_id-1] == self.destination:
                        print(f"Agent {a_id} reached the destination in {t + 1} steps.")
                        self.grid[self.agents[a_id-1]] = 0
                        rest_agents_id_list.remove(a_id)
                        x+=1
        
            if t_agents_id_list ==[]:
                return
        if x==0:
            print("Simulation ended without all agents reaching the destination.")

# Parameters
L = 8  # Grid size
P = 0 # Percentage of obstacles
N = 3  # Number of agents
T = 1000  # Maximum episode length
M = 2  # Agent FoV size

# Create environment and run simulation
env = GridEnvironment(L, P, N, T, M)
print(env.grid)
env.run_simulation()