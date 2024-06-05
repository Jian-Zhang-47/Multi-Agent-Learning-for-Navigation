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
        self.agents_route_dict = {}
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
                    #print(f"Obstacle are {x,y}")
                    break

    def place_destination(self):
        while True:
            x, y = random.randint(0, self.L - 1), random.randint(0, self.L - 1)
            if self.grid[x, y] == 0:
                self.destination = (x, y)  # Destination = 0
                print(f"Destination is at {x,y}")
                break

    def place_agents(self):
        for i in range(self.N):
            while True:
                x, y = random.randint(0, self.L - 1), random.randint(0, self.L - 1)
                if self.grid[x, y] == 0:
                    self.agents_id_list.append(i+1)   # Agent ID
                    self.grid[x, y] =  self.agents_id_list[i]
                    self.agents.append((x, y))
                    self.agents_route_dict[self.agents_id_list[i]] = [self.agents[i]]     # Record the original position of agents
                    break

    def move_agent_multi(self, agent_id):   # Try multiple directions and choose the first one that works
        x, y = self.agents[agent_id-1]
        possible_moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(possible_moves)
        for dx, dy in possible_moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.L and 0 <= ny < self.L and self.grid[nx, ny] == 0:
                self.grid[x, y] = 0
                self.grid[nx, ny] = agent_id
                self.agents[agent_id-1] = (nx, ny)
                self.agents_route_dict[agent_id].append(self.agents[agent_id-1])
                break

    def move_agent_one(self, agent_id):    # Choose only one direction
        x, y = self.agents[agent_id-1]
        possible_moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(possible_moves)
        dx, dy = possible_moves[0]
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.L and 0 <= ny < self.L and self.grid[nx, ny] == 0:
            self.grid[x, y] = 0
            self.grid[nx, ny] = agent_id
            self.agents[agent_id-1] = (nx, ny)
            self.agents_route_dict[agent_id].append(self.agents[agent_id-1])
        else:
            self.agents_route_dict[agent_id].append([x,y])

    def run_simulation(self):
        rest_agents_id_list = self.agents_id_list
        x = 0
        for t in range(self.T):
            t_agents_id_list= rest_agents_id_list
            for a_id in t_agents_id_list:
                if self.agents[a_id-1] == self.destination:
                    print(f"Agent {a_id} reached the destination in 0 steps.")
                    self.grid[self.agents[a_id-1]] = 0
                    rest_agents_id_list.remove(a_id)
                    x+=1

                else:
                    self.move_agent_one(a_id)  # Can choose one or multi trying when agent decides how to move
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
P = 0.3 # Percentage of obstacles
N = 3  # Number of agents
T = 1000  # Maximum episode length
M = 2  # Agent FoV size

# Create environment and run simulation
env = GridEnvironment(L, P, N, T, M)
print("The grid map is:")
print(env.grid)
env.run_simulation()
for i in range(env.N):
    print(f"The Route of Agent {i+1} is: ")
    print(env.agents_route_dict[i+1])
    print()