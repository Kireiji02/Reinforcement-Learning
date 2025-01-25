# Reinforcement Learning Algorithms
Welcome to the Reinforcement Learning Algorithms repository! This project is part of a Year 3 course on reinforcement learning and focuses on implementing and understanding foundational RL techniques. While it serves as a class project, it is designed to be a resource for future learning and experimentation with RL methods.

## Table of contents

- [About the Project](#about-the-project)
- [Implemented Methods](#implemented-methods)
    1. Decaying Epsilon-Greedy
    2. Upper Confidence Bound (UCB)
- [Setup and Usage](#setup-and-usage)

## About the Project
This repository was created as part of a Year 3 class project to explore key algorithms in reinforcement learning. The primary goal is to develop a foundational understanding of RL techniques while building a reusable and extensible codebase for future experimentation.

The project currently includes two classic methods for action selection in multi-armed bandit problems: Decaying Epsilon-Greedy and Upper Confidence Bound (UCB). More algorithms will be added as the project progresses.

## Implemented Methods
1. Decaying Epsilon-Greedy

- Description: Balances exploration and exploitation by gradually reducing the exploration probability (ε) over time. Initially, the agent explores extensively, but as it learns, it focuses on exploiting the best-known actions.

- Use Case: Suitable for static environments where exploration can diminish over time.

- Features:

    - Tracks cumulative rewards for each action over time.

    - Allows visualization of the agent's learning process.

2. Upper Confidence Bound (UCB)

- Description: Balances exploration and exploitation by considering the uncertainty (confidence) in the reward estimates of each action. Actions with higher uncertainty are prioritized for exploration.

- Formula:

    ![argmax_formula](/Asset/RL_argmax_formula.jpg)

- Where:

    - Q(a) : Estimated reward for action .

    - N(a) : Number of times action  has been selected.

    - c : Exploration parameter.

- Use Case: Useful for problems where actions with high uncertainty should be explored more frequently.

## Setup and Usage

### Prerequisites
- Python 3.7 or above
- Libraries:
    - matplotlib
    - numpy

### Installation
Clone this repository:
```
git clone https://github.com/your-username/reinforcement-learning.git
cd reinforcement-learning
```
### Run a Simulation
- For Decaying Epsilon-Greedy:
```
python epsilon_greedy.py
```
- For UCB:
```
python upper_confidence_bound.py
```