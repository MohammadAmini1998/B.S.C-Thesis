Reinforcement Learning in Atari Games
Welcome to my Bachelor's project on Reinforcement Learning (RL) applied to Atari games! 🎮 In this project, I explored the fascinating intersection of deep learning and RL by implementing SARSA and Q-Learning algorithms with deep neural networks.

Overview
The primary objective of this project was to develop an agent capable of learning to play Atari games using RL techniques. To achieve this, I utilized convolutional neural networks (CNNs) to extract features from the raw pixel inputs of Atari game frames. These features were then fed into RL algorithms for training, enabling the agent to learn effective strategies for gameplay.

Implemented Algorithms
SARSA (State-Action-Reward-State-Action)
SARSA is an on-policy RL algorithm that learns to estimate the Q-values of state-action pairs. By updating Q-values based on the observed transitions, the agent gradually improves its policy while interacting with the environment.

The SARSA update equation is given by:

�
(
�
,
�
)
←
�
(
�
,
�
)
+
�
[
�
+
�
�
(
�
′
,
�
′
)
−
�
(
�
,
�
)
]
Q(s,a)←Q(s,a)+α[r+γQ(s 
′
 ,a 
′
 )−Q(s,a)]

Where:

�
(
�
,
�
)
Q(s,a) is the Q-value for state 
�
s and action 
�
a.
�
r is the reward received after taking action 
�
a in state 
�
s.
�
α is the learning rate.
�
γ is the discount factor.
�
′
s 
′
  is the next state.
�
′
a 
′
  is the next action.
Q-Learning
Q-Learning is an off-policy RL algorithm that directly learns the optimal action-value function without requiring a model of the environment. This algorithm updates Q-values based on the maximum expected future reward, allowing the agent to learn optimal strategies through exploration and exploitation.

The Q-Learning update equation is given by:

�
(
�
,
�
)
←
�
(
�
,
�
)
+
�
[
�
+
�
max
⁡
�
′
�
(
�
′
,
�
′
)
−
�
(
�
,
�
)
]
Q(s,a)←Q(s,a)+α[r+γmax 
a 
′
 
​
 Q(s 
′
 ,a 
′
 )−Q(s,a)]

Where:

�
(
�
,
�
)
Q(s,a) is the Q-value for state 
�
s and action 
�
a.
�
r is the reward received after taking action 
�
a in state 
�
s.
�
α is the learning rate.
�
γ is the discount factor.
�
′
s 
′
  is the next state.
Methodology
Preprocessing: Raw pixel inputs from the Atari Gym environments were preprocessed using CNNs to extract meaningful features, facilitating the learning process.
Training: The preprocessed features were then used as inputs to the SARSA and Q-Learning algorithms, which were trained to maximize cumulative rewards over episodes of gameplay.
Evaluation: The trained agents were evaluated on their performance in various Atari games, assessing their ability to achieve high scores and exhibit proficient gameplay.
Implementation Inspiration
This project draws inspiration from the seminal paper "Playing Atari with Deep Reinforcement Learning" by Mnih et al. The methods and techniques presented in the paper served as the foundation for this implementation.

Usage
To replicate or extend this project, follow these steps:

Environment Setup: Set up a Python environment with necessary dependencies, including OpenAI Gym and TensorFlow or PyTorch.
Data Preprocessing: Implement CNN-based preprocessing to extract features from Atari game frames.
Algorithm Implementation: Implement SARSA and/or Q-Learning algorithms with deep neural networks using your preferred deep learning framework.
Training: Train the agents on Atari games by interacting with the environment and updating Q-values based on observed rewards.
Evaluation: Evaluate the trained agents' performance on various Atari games, analyzing their gameplay and score achievements.
Contributions and Feedback
## Before Training
![Before training](https://user-images.githubusercontent.com/49214384/216965955-187c2743-c680-4907-9a95-bacb452f236c.gif)

## After training with DQM
![Train with DDQN](https://user-images.githubusercontent.com/49214384/216972988-674a1627-199f-4848-a95b-5fa087eb578e.gif)


## After training with Deep SARSA
![Train wih Deep SARSA](https://user-images.githubusercontent.com/49214384/216966067-48b3fa63-41e2-4ee7-b100-675c6fd2fe49.gif)
