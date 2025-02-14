import matplotlib.pyplot as plt
import numpy as np
import random

#------------------------------------Classes construction------------------------------------#

class Bandit:
    def __init__(self, n, r):
        self.n = n
        self.reward = r[0]
        self.weight = r[1]
     
class Agent:
    def __init__(self,a,t):
        self.action = a # Bandit class
        self.q_value = [0.0 for _ in range(len(a))]
        self.action_value = [[0,0] for _ in range(len(a))] # [action count, cumulative reward]
        self.step = t
        self.epsilon = 1.0
        self.decay_rate = 4.0 # Around 2.0 - 5.0
        self.mode = 0 # 0 = exploration, 1 = exploitation

        #for output
        self.simulated_rewards = [] # [action, recieved reward, mode]
        self.cumulative_rewards = [[0] for _ in range(len(a))]  # Track cumulative reward for each bandit
        self.epsilon_values = [self.epsilon]  # Store epsilon values over time

    def normalize_q(self, q):
        normal = []
        for count,cumulative_reward in q:
            if count != 0:
                normal.append(cumulative_reward/count)
            else:
                normal.append(0)

        total = sum(normal)
        if total != 0:
            return [p / total for p in normal]
        else:
            return [1.0 / len(q) for _ in q]

    def simulate(self):
        for _ in range(self.step):
            # Decaying epsilon value
            self.epsilon = max(0.1, self.epsilon * (1.0 - (10 ** - self.decay_rate)))
            self.epsilon_values.append(self.epsilon)
            self.mode = 0 if random.random() < self.epsilon else 1

            # Select an action based on q_value weight at timestep t
            if self.mode == 0:
                action_t = random.choices(
                    self.action, 
                    weights= [1.0/len(self.action) for _ in range(len(self.action))], 
                    k=1
                )[0] 
            else:
                action_t = self.action[max(range(len(self.q_value)), key=lambda i: self.q_value[i])]

            # Return a reward for that action at timestep t and mode
            reward = random.choices([0, action_t.reward], [1.0 - action_t.weight, action_t.weight], k=1)[0]
            self.simulated_rewards.append([action_t.n, reward, self.mode])

            # Update cumulative rewards for the selected bandit
            self.cumulative_rewards[action_t.n].append(self.cumulative_rewards[action_t.n][-1] + reward)

            # For other bandits, copy the last cumulative reward (no change)
            for i in range(len(self.cumulative_rewards)):
                if i != action_t.n:
                    self.cumulative_rewards[i].append(self.cumulative_rewards[i][-1])

            
            # Count actions and add a reward to the action value
            self.action_value[action_t.n] = [x+y for x,y in zip(self.action_value[action_t.n],[1,self.simulated_rewards[-1][1]])]

            # Normalize
            self.q_value = self.normalize_q(self.action_value)

        return self.simulated_rewards, self.q_value
        
#------------------------------------Running the script------------------------------------#

if __name__ == "__main__":

    #------------------------------------Initilization------------------------------------#
    steps = 10000 # Steps per iteration
    iterations = 10

    reward_matrix = [[5,0.9],[100,0.11],[50,0.2],[1,1]] # [reward, probability]
    # reward_matrix = [[1000,0.01],[500,0.02],[100,0.1],[50,0.2],[20,0.5]]

    bandits = [Bandit(i, reward_matrix[i]) for i in range(len(reward_matrix))]
    agent = Agent(bandits,steps)

    weight = [1.0 / len(bandits) for _ in range(len(bandits))] # Pre-initialize for edge cases (for output only not computed)

    #--------------------------------------Simulation--------------------------------------#

    for i in range(iterations):
        result, weight = agent.simulate()

        #----------------------------------------Result----------------------------------------#

    action_counts = [val[0] for val in agent.action_value]
    average_rewards = [val[1] / val[0] if val[0] > 0 else 0 for val in agent.action_value]

    print("\nSummary:")
    for i, (count, avg_reward) in enumerate(zip(action_counts, average_rewards)):
        print(f"Action {i + 1}: Count = {count}, Average Reward = {avg_reward:.2f}, Weight = {weight[i]:.4f}")
    print('-------------------------------------------')

     # Create a single figure with two subplots (stacked vertically)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot cumulative reward for each bandit on the first subplot
    for i, cumulative_reward in enumerate(agent.cumulative_rewards):
        ax1.plot(range(len(cumulative_reward)), cumulative_reward, label=f"Bandit {i + 1}")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Cumulative Reward")
    ax1.set_title("Cumulative Reward for Each Bandit Over Time")
    ax1.legend()
    ax1.grid(True)

    # Plot epsilon decay on the second subplot
    ax2.plot(range(len(agent.epsilon_values)), agent.epsilon_values, label="Epsilon Decay", color='red', linestyle='dashed')
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Epsilon Value")
    ax2.set_title("Epsilon Decay Over Time")
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()