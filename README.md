# Reinforcement Learning in Atari Games

Welcome to my Bachelor's project on Reinforcement Learning (RL) applied to Atari games! 🎮 In this project, I explored the fascinating intersection of deep learning and RL by implementing SARSA and Q-Learning and Double Q-Learning algorithms with deep neural networks.

## Overview

The primary objective of this project was to develop an agent capable of learning to play Atari games using RL techniques. To achieve this, I utilized convolutional neural networks (CNNs) to extract features from the raw pixel inputs of Atari game frames. These features were then fed into RL algorithms for training, enabling the agent to learn effective strategies for gameplay.

## Implemented Algorithms

### SARSA (State-Action-Reward-State-Action)

SARSA is an on-policy RL algorithm that learns to estimate the Q-values of state-action pairs. By updating Q-values based on the observed transitions, the agent gradually improves its policy while interacting with the environment.

The SARSA update equation is given by:

$\ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right] \$

Where:
- $Q(s, a)$ is the Q-value for state $s$ and action $a$.
- $r$ is the reward received after taking action $a$ in state $s$.
- $\alpha$ is the learning rate.
- $\gamma$ is the discount factor.
- $s'$ is the next state.
- $a'$ is the next action.

### Q-Learning

Q-Learning is an off-policy RL algorithm that directly learns the optimal action-value function without requiring a model of the environment. This algorithm updates Q-values based on the maximum expected future reward, allowing the agent to learn optimal strategies through exploration and exploitation.

The Q-Learning update equation is given by:

$\ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right] \$

Where:
- $Q(s, a)$ is the Q-value for state $s$ and action $a$.
- $r$ is the reward received after taking action $a$ in state $s$.
- $\alpha$ is the learning rate.
- $\gamma$ is the discount factor.
- $s'$ is the next state.
### Double Deep Q-Networks (Double DQN)

In addition to SARSA and Q-Learning, this project also explores the concept of Double Deep Q-Networks (Double DQN). Double DQN is an enhancement to the original Deep Q-Network (DQN) algorithm introduced by DeepMind.

Traditional DQN algorithms tend to overestimate Q-values, leading to suboptimal policies during training. Double DQN addresses this issue by decoupling action selection from value estimation, effectively reducing overestimation bias.

The key idea behind Double DQN is to use one set of parameters to select actions (the "online" network) and another set to evaluate those actions (the "target" network). By periodically updating the target network's parameters to match those of the online network, Double DQN mitigates overestimation bias and improves training stability.

The Double Q-Learning update equation is given by:

$\ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q\left(s', \arg\max_{a'} Q(s', a'; \theta_{\text{online}}); \theta_{\text{target}}\right) - Q(s, a) \right] \$

Where:
- $Q(s, a)$ is the Q-value for state $ s $ and action $ a $.
- $r$ is the reward received after taking action $a$ in state $s$.
- $\alpha$ is the learning rate.
- $\gamma$ is the discount factor.
- $s'$ is the next state.
### Benefits

- **Reduced Overestimation Bias**: Double DQN reduces the tendency of traditional DQN algorithms to overestimate Q-values, resulting in more accurate value estimates and improved learning.
- **Improved Training Stability**: By mitigating overestimation bias, Double DQN leads to more stable and reliable training, ultimately yielding better performance on challenging tasks.

## Methodology

1. **Preprocessing**: Raw pixel inputs from the Atari Gym environments were preprocessed using CNNs to extract meaningful features, facilitating the learning process.
   
2. **Training**: The preprocessed features were then used as inputs to the SARSA and Q-Learning algorithms, which were trained to maximize cumulative rewards over episodes of gameplay.

3. **Evaluation**: The trained agents were evaluated on their performance in Pong Atari games, assessing their ability to achieve high scores and exhibit proficient gameplay. Please note that the algorithms can be applied in any environment.

## Results

### Performance Comparison

The performance of the SARSA, DQN, and Double DQN algorithms was evaluated on the Pong environment. After extensive testing, it was observed that Double DQN outperformed both Deep SARSA and DQN in terms of learning efficiency and final performance metrics.

### Observations

- **Double DQN**: The Double DQN algorithm exhibited superior performance compared to Deep SARSA and DQN. It achieved higher scores and learned more efficiently, showcasing its effectiveness in learning complex strategies for playing the Pong game.
  
- **Deep SARSA**: While Deep SARSA showed promising results, it was outperformed by Double DQN. However, it demonstrated competitive performance and could be further optimized for better results in future experiments.
  
- **DQN**: The original DQN algorithm also showed reasonable performance, but it lagged behind Double DQN and Deep SARSA in terms of learning speed and final performance on the Pong environment.
### Simulation's result
![reward](https://github.com/MohammadAmini1998/B.S.C-Thesis/assets/49214384/936082eb-5a7e-459c-81e7-8922b2dde5db)
**Figure:** The y-axis represents the reward, and the x-axis represents the number of episodes.


## Implementation Inspiration

This project draws inspiration from the seminal paper ["Playing Atari with Deep Reinforcement Learning"](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) by Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. The methods and techniques presented in the paper served as the foundation for this implementation.

## Usage

To replicate or extend this project, simply clone the project and run main.py.

## Contributions and Feedback

I welcome contributions, feedback, and suggestions for improvement! If you have any ideas, questions, or insights regarding this project, feel free to reach out and engage in discussions.

Let's continue exploring the exciting possibilities of Reinforcement Learning together! 🚀
