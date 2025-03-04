# Reinforcement Learning Algorithms
Welcome to the Reinforcement Learning Algorithms repository! This project is part of a Year 3 course on reinforcement learning and focuses on implementing and understanding foundational RL techniques. While it serves as a class project, it is designed to be a resource for future learning and experimentation with RL methods.

## Table of contents

- [About the Project](#about-the-project)
- [Implemented Methods](#implemented-methods)
- [Setup and Usage](#setup-and-usage)
- [Result Evaluation](#result-evaluation)

## About the Project
This repository was created as part of a Year 3 class project to explore key algorithms in reinforcement learning. The primary goal is to develop a foundational understanding of RL techniques while building a reusable and extensible codebase for future experimentation.

The project currently includes two classic methods for action selection in multi-armed bandit problems: Decaying Epsilon-Greedy and Upper Confidence Bound (UCB). More algorithms will be added as the project progresses.

## Implemented Methods
i. Decaying Epsilon-Greedy

- Description: Balances exploration and exploitation by gradually reducing the exploration probability (ε) over time. Initially, the agent explores extensively, but as it learns, it focuses on exploiting the best-known actions.

- Use Case: Suitable for static environments where exploration can diminish over time.

- Features:

    - Tracks cumulative rewards for each action over time.

    - Allows visualization of the agent's learning process.

ii. Upper Confidence Bound (UCB)

- Description: Balances exploration and exploitation by considering the uncertainty (confidence) in the reward estimates of each action. Actions with higher uncertainty are prioritized for exploration.

- Formula:

$$
a_t = \arg \max_a \left( Q(a) + c \sqrt{\frac{\log(t)}{N(a)}} \right)
$$

- Where:

    - Q(a) : Estimated reward for action .

    - N(a) : Number of times action  has been selected.

    - t : the current timestep.

    - c : Exploration parameter.

- Use Case: Useful for problems where actions with high uncertainty should be explored more frequently.

## Setup and Usage

### Prerequisites
- Python 3.7 or above
- Libraries:
    - matplotlib
    - random (part of Python's standard library)

### Installation
Clone this repository:
```
git clone https://github.com/Kireiji02/reinforcement-learning.git
cd reinforcement-learning
```
### Run a Simulation
- For Decaying Epsilon-Greedy:

Parameters to change: `steps, iterations, decay_rate and reward matrix`
```py
steps = 10000 #steps per iteration
iterations = 10
decay_rate = 4.0 # Around 2.0 - 5.0
reward_matrix = [[5,0.9],[100,0.1],[50,0.2],[1,1]] # [reward, probability]
```

- For UCB:

Parameters to change: `steps, iterations, explore_rate and reward matrix`
```py
steps = 10000 #steps per iteration
iterations = 10
explore_rate = 2
reward_matrix = [[5,0.9],[100,0.1],[50,0.2],[1,1]] # [reward, probability]
```

## Result Evaluation

### Decaying Epsilon Greedy

To understand what epsilon is, it's essentially a parameter that determines whether to explore or exploit the results. The agent randomly selects a number between 0 and 1. If the result is greater than or equal to epsilon, it sets the variable `self.mode` to **1 (exploitation)**; otherwise, it sets `self.mode` to **0 (exploration)**.

```py
self.mode = 0 if random.random() < self.epsilon else 1
```

If `self.mode` is set to **0**, the agent will randomly choose between all actions without bias. In contrast, if `self.mode` is set to **1**, it will select the action with the highest action value (Weight / Average reward). All other actions will be treated equally, regardless of their value differences.

```py
if self.mode == 0:
                action_t = random.choices(
                    self.action, 
                    weights= [1.0/len(self.action) for _ in range(len(self.action))], 
                    k=1
                )[0] 
            else:
                action_t = self.action[max(range(len(self.q_value)), key=lambda i: self.q_value[i])]

```

The epsilon value decays exponentially according to the equation below. With 100,000 timesteps across 10 iterations, decay rates between 2.0 (very fast decay) and 5.0 (gradual decay over the timesteps) produce visible changes.

```py
self.decay_rate = 4.0 # Around 2.0 - 5.0
self.epsilon = max(0.1, self.epsilon * (1.0 - (10 ** - self.decay_rate)))
```

The reward matrix may seem unconventional. Here’s the breakdown: it represents the number of actions (bandits), where each action has two parameters. The first parameter is the reward value, which is not limited to 1 or 0 but can be any arbitrary number. The second parameter is the probability of receiving that reward. This means that different actions can have the same expected value; for example, an action with a reward of 100 at a 10% probability is equivalent to another with a reward of 50 at a 20% probability (1.0 is 100%).

```py
reward_matrix = [[5,0.9],[100,0.11],[50,0.2],[1,1.0]] # [reward, probability]
```

From the reward matrix above, we can determine that Action 2 has the highest expected value mathematically. Here are some example scenarios:

![Decay_rate=2.0](/assets/DEG_Decay_rate_2.0.jpg)

**Decay rate = 2.0**

In this image, the decay rate is 2.0, causing epsilon to quickly saturate at 0.1 around 3,000 timesteps. At this point, the agent primarily exploits the action it perceives as the best while exploring the other options only 10% of the time.

However, at 20,000 timesteps, the agent realizes that Action 2 yields slightly more reward than Action 3 and correctly shifts its exploitation strategy. This decay rate is too fast, as the most rewarding action appears only 10% of the time.

As a result, the agent may prematurely settle on a suboptimal action, missing the possibility that an action with slightly lower potential reward but a higher probability of success could have been identified at earlier timesteps.

---

![Decay_rate=5.0](/assets/DEG_Decay_rate_5.0.jpg)

**Decay rate = 5.0**

In contrast, this slower decay rate did not even reach the 0.1 threshold at 100,000 timesteps, resulting in a more exploratory behavior with no significant exploitation until the very late timesteps.

---

![Decay_rate=4.0](/assets/DEG_Decay_rate_4.0.jpg)

**Decay rate = 4.0**

Here's my take on one of the **possibly more optimal** decay rates for this specific set of deterministic timesteps and rewards. The agent has sufficient time to explore, but not excessively. The balance in this graph is clearly visible.

---

### Upper Confidence Bound

This approach is similar to epsilon-greedy, but instead of using epsilon to decide whether to explore or exploit, it employs a compensation value that increases over time if an action is not selected for an extended period. This ensures that some actions will still be explored at later timesteps, even if their initial selection probability was very low.

```py
# Update the uppder confidence bound
for i in range(len(self.action)):
    if self.action_value[i][0] == 0:
        self.upper_confidence[i] = max(a.reward for a in self.action) #buffer
    else:
        self.upper_confidence[i] = math.sqrt(math.log(t+1) / 
        (2 * self.action_value[i][0])) * self.explore_gain
```

The exploration rate of this method is entirely determined by the explore_rate parameter (c).

![Explore_rate=1](/assets/UCB_Exploration_rate_1.jpg)

**Exploaration rate = 1**

This value results in very little exploration and immediately exploits the first selected action, making it unusable.

---

![Explore_rate=3](/assets/UCB_Exploration_rate_3.jpg)

**Exploaration rate = 3**

This always produces the correct answer, but the exploration of the next best value is excessive, making it suboptimal.

---

![Explore_rate=10](/assets/UCB_Exploration_rate_10.jpg)

**Exploaration rate = 10**

This continues the trend of excessive exploration.

---

![Explore_rate=1.5](/assets/UCB_Exploration_rate_1.5.jpg)

**Exploaration rate = 1.5**

An exploration value of 1.5 is the most optimal for the current reward matrix and timesteps.

### Summary

Decaying Epsilon-Greedy (DEG) and Upper Confidence Bound (UCB) are two exploration strategies used in multi-armed bandit problems and reinforcement learning. DEG relies on an epsilon parameter that decreases over time, shifting the agent from an exploration phase, where it selects actions randomly, to an exploitation phase, where it chooses the best-known action. This approach is simple and effective but requires careful tuning of the decay rate to avoid premature exploitation or excessive exploration. In contrast, UCB uses a confidence-based approach, assigning an exploration bonus to actions that have been chosen less frequently, ensuring they are periodically revisited. This method balances exploration and exploitation dynamically, making it well-suited for environments where rewards change over time. While DEG provides a straightforward way to control exploration, UCB optimally selects actions based on uncertainty, often leading to more efficient learning.