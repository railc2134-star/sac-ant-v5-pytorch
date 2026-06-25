Soft Actor-Critic Agent for Ant-v5 (Continuous Control)

This project implements a deep reinforcement learning agent to solve the Ant-v5 environment using a Soft Actor-Critic (SAC)-inspired approach.

The agent learns a stochastic policy for high-dimensional continuous control tasks.

Environment

- Gymnasium Ant-v5
- High-dimensional state space (robot locomotion)
- Continuous multi-joint action space

Model Architecture

Actor:
- Fully connected neural network
- Outputs mean and standard deviation for Gaussian policy
- Samples actions using reparameterization trick
- Applies tanh squashing for bounded actions

Critics:
- Two Q-networks (twin critic architecture)
- Estimate Q(s, a) for stability
- Target networks used for soft updates

Training Method

- Experience replay buffer
- Off-policy learning
- Soft value estimation
- Twin Q-learning (Clipped Double Q-learning idea)
- Entropy regularization for exploration
- Soft target network updates

Key Concepts

- Soft Actor-Critic (SAC)-style training
- Continuous action reinforcement learning
- Stochastic policy optimization
- Entropy maximization
- Twin Q-value estimation

Limitations

This implementation is a simplified SAC-like system:

- No automatic entropy tuning
- No true SAC target value formulation consistency
- Simplified log-prob correction for tanh squashing
- No vectorized environments
