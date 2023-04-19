# Reinforcement Learning for Self-Driving Cars - DQN Automated Driving
## Introduction

Reinforcement learning has made the development of self-driving cars easier by enabling agents to better understand their environment and take appropriate actions. In this project, we explore how deep Q-learning can be used to develop self-driving cars.

## Deep Q-Learning
Before the agent transitions to a new state, it must know the Q(S,A), the Q value of the state when the agent is moving up. S represents the current state, and A represents the action taken by the agent. After the transition, the new Q value becomes R(S,A)+γ〖max〗(a^' ) Q(S^',A^' ), where R is the reward function and γ is the discount factor. The temporal difference is given by R(S,A)+γ〖max〗(a^' ) Q(S^',A^' )-Q(S,A). Both the terms R(S,A)+γ〖max〗_(a^' ) Q(S^',A^' ) and Q(S,A) should be equal so that the agent knows everything about the environment. We use neural networks to predict Q values based on possible actions in the environment. These Q values are compared with the Q target, and the loss function is calculated by ∑▒〖(Qtarget-Q)〗^2. We then use backpropagation to update the parametric data (weight and bias) and schedule the agent's movements using softmax activation.

## Experience Replay
In real-world environments, the agent's next state is often correlated with its previous states. To handle this, we use experience replay. The agent takes batches of experiences from its previous states, which are stored in its memory. These batches are selected through uniform distribution experiences and are used to train the agent to better handle rare experiences, such as sharp edges where it has limited experience. Experience replay also enables us to train the agent more quickly by replaying experiments rather than redoing them.

By using deep Q-learning and experience replay, we can develop better self-driving cars that are better equipped to handle complex environments.

## Architecture Diagram
![image](https://user-images.githubusercontent.com/83855692/233080796-5ee69cc2-3681-4321-a33f-31749940c7cc.png)
![image](https://user-images.githubusercontent.com/83855692/233080823-537f40b7-737c-43a9-9488-3441fa1a7532.png)

## Sample Output
![image](https://user-images.githubusercontent.com/83855692/193855434-f75dde65-fa97-4747-9c1a-55418c92cdb7.png)

## Loss Graph in Deep Q Learning
![image](https://user-images.githubusercontent.com/83855692/193855553-6e50a874-af6c-44c0-b9c9-87638834baca.png)
