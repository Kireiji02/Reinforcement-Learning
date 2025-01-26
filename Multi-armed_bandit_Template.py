import random

#------------------------------------Classes construction------------------------------------#

class Bandit:
    def __init__(self, n, r):
        self.n = n
        self.reward = r[0]
        self.weight = r[1]

class Agent:
    def __init__(self,a,t):
        self.action = a
        self.policy = [1.0/len(a) for i in range(len(a))]
        self.step = t

    def simulate(self):
        simulated_rewards = [] # [action, recieved reward]
        for i in range(self.step):
            actions = random.choices(self.action, weights=self.policy, k=1)[0]
            simulated_rewards.append([actions.n,random.choices([0,actions.reward], [1.0-actions.weight, actions.weight], k=1)[0]])

        return simulated_rewards
        
#------------------------------------Running the script------------------------------------#

if __name__ == "__main__":

    #------------------------------------Initilization------------------------------------#

    steps = 10

    reward_matrix = [[5,0.9],[100,0.1],[50,0.2]] # [reward, probability]
    # reward_matrix = [[1000,0.01],[500,0.02],[100,0.1],[50,0.2],[20,0.5]]

    bandits = [Bandit(i, reward_matrix[i]) for i in range(len(reward_matrix))]
    agent = Agent(bandits,steps)

    #--------------------------------------Simulation--------------------------------------#
    
    result = agent.simulate()

    #----------------------------------------Result----------------------------------------#

    for idx,x in enumerate(result):
        print(f'Timestep{idx+1}: Action = {x[0]}, Reward = {x[1]}')