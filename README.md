# Cooperative Multi-Agent Learning for Navigation

Field-of-View (FoV)-based navigation is a task in which a group of agents must reach an unknown destination relying solely on their local observations. Each agent has limited visibility, being able to perceive only a small portion of the surrounding environment. This project is a partial implementation and customization of FoV-based navigation scheme introduced in [1], following this procedure: 

## 1.Setup
Develop an object-oriented Python code to simulate the navigation of multiple agents in a grid environment with random obstacles and a single random destination [2] (similar to Fig. 1, left side), based on the following assumptions: 
- a) Agents can move randomly one step at a time in four main directions during each time slot. 
- b) Agents are restricted from moving over obstacles, beyond the grid borders, or into the locations occupied by other agents. In such cases of invalid actions, they should remain stationary. 
- c) The code should accept the following parameters as input: 
  - Grid size (𝐿) 
  - Number of obstacles (as a percentage to grid size) (𝜚) 
  - Number of agents (𝑁) 
  - Maximum episode length (𝑇) 
  - Agent FoV size (applicable for next section) (𝑀) 
- d) The  code  should  track  and  log  the  trajectories  of  agents  until  either  all  agents successfully reach the destination or the maximum episode length is reached. 

![](Aspose.Words.8898c72f-1f43-4d2d-bc24-63a25ca5338d.001.jpeg)

Fig. 1 – Example snapshot of the simulation environment and agents’ states [1] 

## 2.Independent RL agents
Based on the simulation environment in the previous step, develop and train *individual single-agent RL algorithms for each agent* with the following assumptions: 
- a) For each agent, the action space is {up, down, left, right}, and the state space is constrained to agent’s FoV as per the system model in reference [1] (similar to Fig. 1, right side). A reward of +1 is given for reaching the destination. 
- b) Since the only contact point among agents is that they cannot occupy the same grid slot, the algorithms are expected to converge rapidly, especially for small grids, e.g., 𝐿 < 10 and 𝜚 < 10%. 
- c) Please utilize the D3QL algorithm proposed in [3] to train agents. 
## 3.Centralized Training Distributed Execution (CTDE)
Now, we assume that agents transmit their observations to a central entity (e.g., a cloud server) via a perfect channel after each move. The server aggregates the local states of the agents and utilizes this information to train the agents' local models. While the agents make decisions based on their individual observations (as in the previous step), the utilization of collective information during training suggests that the CTDE setup is expected to outperform the Distributed setup. 
- a) Please utilize the same D3QL algorithm proposed in [3] to complete this stage as well. 

- b)  It  is  preferred  to  have  separate  Python  classes  for  the  central  entity,  agents, environment, configuration, and learning algorithm. 

## References 

[1]. Abdel-Aziz,  Mohamed  K.,  et  al.  "Cooperative  Multi-Agent  Learning  for  Navigation  via  Structured  State Abstraction." IEEE Transactions on Communications (2024).

[2]. Lin, Toru, et al. "Learning to ground multi-agent communication with autoencoders." Advances in Neural Information Processing Systems 34 (2021): 15230-15242. 

   Code:[ https://github.com/kandouss/marlgrid ](https://github.com/kandouss/marlgrid)

[3]. Mazandarani, Hamidreza, Masoud Shokrnezhad, and Tarik Taleb. "A Semantic-Aware Multiple Access Scheme for Distributed, Dynamic 6G-Based Applications." *arXiv preprint arXiv:2401.06308* (2024).

   Link:[ https://arxiv.org/html/2401.06308v1 ](https://arxiv.org/html/2401.06308v1)
