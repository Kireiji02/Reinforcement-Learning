# Reinforcement Learning Algorithms
Welcome to the Reinforcement Learning Algorithms repository! This project is part of a Year 3 course on reinforcement learning and focuses on implementing and understanding foundational RL techniques. While it serves as a class project, it is designed to be a resource for future learning and experimentation with RL methods.

## Table of contents

- [About the Project](#about-the-project)
- [Implemented Methods](#implemented-methods)
    i. Decaying Epsilon-Greedy
    ii. Upper Confidence Bound (UCB)
- [Setup and Usage](#setup-and-usage)

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

Parameters to change: `steps, iterations and reward matrix`
```py
steps = 10000 #steps per iteration
iterations = 10
reward_matrix = [[5,0.9],[100,0.1],[50,0.2],[1,1]] # [reward, probability]
```

- For UCB:
```
python Upper_confidence_bound.py
```