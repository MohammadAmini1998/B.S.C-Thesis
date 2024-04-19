# Reinforcement Learning in Atari Games

Welcome to my Bachelor's project on Reinforcement Learning (RL) applied to Atari games! 🎮 In this project, I explored the fascinating intersection of deep learning and RL by implementing SARSA and Q-Learning algorithms with deep neural networks.

## Overview

The primary objective of this project was to develop an agent capable of learning to play Atari games using RL techniques. To achieve this, I utilized convolutional neural networks (CNNs) to extract features from the raw pixel inputs of Atari game frames. These features were then fed into RL algorithms for training, enabling the agent to learn effective strategies for gameplay.

## Implemented Algorithms

### SARSA (State-Action-Reward-State-Action)

SARSA is an on-policy RL algorithm that learns to estimate the Q-values of state-action pairs. By updating Q-values based on the observed transitions, the agent gradually improves its policy while interacting with the environment.

The SARSA update equation is given by:

$\Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]\$

Where:
- $Q(s, a) \$ is the Q-value for state $\( s \)$ and action $\( a \)$.
- $\r \$ is the reward received after taking action $\( a \)$ in state $\( s \)$.
- $\\alpha \$ is the learning rate.
- $\\gamma \$ is the discount factor.
- $\s' \$ is the next state.
- $\a' \$ is the next action.

### Q-Learning

Q-Learning is an off-policy RL algorithm that directly learns the optimal action-value function without requiring a model of the environment. This algorithm updates Q-values based on the maximum expected future reward, allowing the agent to learn optimal strategies through exploration and exploitation.

The Q-Learning update equation is given by:

$\ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right] \$

Where:
- $\( Q(s, a) \) is the Q-value for state \( s \) and action \( a \)$.
- $\( r \) is the reward received after taking action \( a \) in state \( s \)$.
- $\( \alpha \) is the learning rate$.
- $\( \gamma \) is the discount factor$.
- $\( s' \) is the next state$.

## Methodology

1. **Preprocessing**: Raw pixel inputs from the Atari Gym environments were preprocessed using CNNs to extract meaningful features, facilitating the learning process.
   
2. **Training**: The preprocessed features were then used as inputs to the SARSA and Q-Learning algorithms, which were trained to maximize cumulative rewards over episodes of gameplay.

3. **Evaluation**: The trained agents were evaluated on their performance in various Atari games, assessing their ability to achieve high scores and exhibit proficient gameplay.

## Implementation Inspiration

This project draws inspiration from the seminal paper ["Playing Atari with Deep Reinforcement Learning"](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) by Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. The methods and techniques presented in the paper served as the foundation for this implementation.

## Usage

To replicate or extend this project, follow these steps:

1. **Environment Setup**: Set up a Python environment with necessary dependencies, including OpenAI Gym and TensorFlow or PyTorch.
   
2. **Data Preprocessing**: Implement CNN-based preprocessing to extract features from Atari game frames.
   
3. **Algorithm Implementation**: Implement SARSA and/or Q-Learning algorithms with deep neural networks using your preferred deep learning framework.
   
4. **Training**: Train the agents on Atari games by interacting with the environment and updating Q-values based on observed rewards.
   
5. **Evaluation**: Evaluate the trained agents' performance on various Atari games, analyzing their gameplay and score achievements.

## Contributions and Feedback

I welcome contributions, feedback, and suggestions for improvement! If you have any ideas, questions, or insights regarding this project, feel free to reach out and engage in discussions.

Let's continue exploring the exciting possibilities of Reinforcement Learning together! 🚀
