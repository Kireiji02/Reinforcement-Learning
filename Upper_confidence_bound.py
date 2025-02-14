import matplotlib.pyplot as plt
import random
import math

#------------------------------------Classes construction------------------------------------#

class Bandit:
    def __init__(self, n, r):
        self.n = n
        self.reward = r[0]
        self.weight = r[1]

class Agent:
    def __init__(self, a, t, c):
        self.action = a # Bandit class
        self.q_value = [1.0/len(a) for _ in range(len(a))]
        self.action_value = [[0,0] for _ in range(len(a))] # [action count, cumulative reward]
        self.step = t
        self.upper_confidence = [1.0 for _ in range(len(a))]
        self.explore_gain = c # exploration gain 

        #for output
        self.simulated_rewards = [] # [action, recieved reward]
        self.cumulative_rewards = [[0] for _ in range(len(a))]  # Track cumulative reward for each bandit

    def normalize(self, val):
        total = sum(val)
        if total != 0:
            return [p / total for p in val]
        else:
            return [1.0 / len(val) for _ in val]

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
        for t in range(self.step):
            
            # Select an action based on q_value weight at timestep t
            action_t = self.action[max(range(len(self.q_value)), key=lambda i: self.q_value[i])]

            # Update the uppder confidence bound
            for i in range(len(self.action)):
                if self.action_value[i][0] == 0:
                    self.upper_confidence[i] = max(a.reward for a in self.action) #buffer
                else:
                    self.upper_confidence[i] = math.sqrt(math.log(t+1) / (2 * self.action_value[i][0])) * self.explore_gain

            # Return a reward for that action at timestep t and mode
            reward = random.choices([0, action_t.reward], [1.0 - action_t.weight, action_t.weight], k=1)[0]
            self.simulated_rewards.append([action_t.n, reward])

            # Update cumulative rewards for the selected bandit
            self.cumulative_rewards[action_t.n].append(self.cumulative_rewards[action_t.n][-1] + reward)

            # For other bandits, copy the last cumulative reward (no change)
            for i in range(len(self.cumulative_rewards)):
                if i != action_t.n:
                    self.cumulative_rewards[i].append(self.cumulative_rewards[i][-1])

            # Count actions and add a reward to the action value
            self.action_value[action_t.n] = [x+y for x,y in zip(self.action_value[action_t.n],[1,self.simulated_rewards[-1][1]])]

            # Normalize
            self.q_value = self.normalize([q + u for q, u in zip(self.normalize_q(self.action_value), self.upper_confidence)])

        return self.simulated_rewards, self.q_value
        
#------------------------------------Running the script------------------------------------#

if __name__ == "__main__":

    #------------------------------------Initilization------------------------------------#

    steps = 10000 # Steps per iteration
    iterations = 10
    explore_rate = 1.5

    reward_matrix = [[5,0.9],[100,0.11],[50,0.2],[1,1.0]] # [reward, probability]
    # reward_matrix = [[1000,0.01],[500,0.02],[100,0.1],[50,0.2],[20,0.5]]

    bandits = [Bandit(i, reward_matrix[i]) for i in range(len(reward_matrix))]
    agent = Agent(bandits,steps,explore_rate)

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

    for i, cumulative_reward in enumerate(agent.cumulative_rewards):
            plt.plot(range(len(cumulative_reward)), cumulative_reward, label=f"Bandit {i + 1}")

    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward for Each Bandit Over Time - Upper Confidence Bound")
    plt.legend()
    plt.show()