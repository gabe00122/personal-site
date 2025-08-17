---
title: Online transformer RL
description: Training a transformer with PPO to solve POMDP's
date: '2025-7-26'
categories:
  - reinforcement learning
  - jax
  - flax
  - transformer
  - deep learning
published: false
---

The goal for this project was to see if a simple RL policy gradient could reliably train a transformer from scratch to solve environments to require good context utilization. Solving environments with long context or variable length context was out of scope although I have some thoughts on how those could be handled too.

These results show that transformers can be trained to utilize context in a reasonable amount of time on consumer hardware with on policy reinforcement learning.

## Approach
The core idea behind this implementation is to treat each trajectory as the atomic unit of learning. Because the context influences the action selection training with only part of a trajectory would be off policy.

Actions are inferenced in parallel over multiple agents for computational efficiency and and stability.

When an entire trajectory is collected for each agent a update step is preformed.

Each update step consists of multiple epochs over the rollout data.
For each update the rollout is shuffled and split into mini-batches.
Each mini batch is based back through the network using a causal mask and for a single gradient step.

After the rollout is collected the advantage is calculated using Generalized Advantage Estimators
For each epoch over the rollout this batch of trajectories is shuffled and split into multiple mini-batches.
Each minibatch uses a standard PPO loss function.



## Architecture

The shape of the input observations is [agent, sequence, ...dims]
* Agents this is [environment, agent] flattened where environment is a copy of the environment and agent is one of multiple possible agents in each environment.
* sequence is the length of the trajectory to pass to the model, for inference where a KV cache is used this should be size 1. For back propagation after a rollout is collected this should be the size of the full trajectory.
* dims is environment specific observation dimensions

As well as observation, the last action and reward are passed to the model (these can be considered part of the observation)


This model uses three embedding layers:
* Last action
  * This is similar to a tied weight embedding for a LLM and is also used as the actor head
* Last reward
  * This is a simple linear projection of last reward
* Observation
  * Environment specific, for grid environments this is a simple cnn

All three embeddings are simply added together and feed into the next layers.

This model uses multiple transformer layers, each transformer layer uses a attention block and a feed forward block
The transformer layers use pre-norm, rope positional embeddings and grouped query attention with optional sliding window attention.

The model then uses the output from the transformer layers for the action head (the transposed action embedding) and a value function. I found that preformace is slightly better of the value function uses an additional hidden layer and activation as opposed to being just a linear transformer.


## Performance
A small transformer (2m parameters) can train at 2 million environment steps per second on a single RTX 5090

## Limitations
The obvious limitation of this approach is it won't work for continuous RL, the context size will always be finite and requiring to make a update on the entire context will lower the frequency of model updates as the trajectories get larger.


## Results

A single way to test memory/context is to learn a 2d grid environment. A new grid is generated each episode and agents need to find a random location within this grid. When they get to that location they receive a reward and are moved to another random location. They only way to achieve more rewards is to explore effectively and remember your way back based on features in the random grid.

Observations are given as a small grid view of one hot tile types, a small cnn is used to encode the observations.
