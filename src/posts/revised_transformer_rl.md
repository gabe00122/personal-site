---
title: Online Transformer RL
description: Training a transformer with PPO to solve POMDPs
date: '2025-07-26'
categories:
  - reinforcement learning
  - jax
  - flax
  - transformer
  - deep learning
published: false
---

The goal of this project was to see if a simple RL policy gradient method could train a transformer from scratch to solve partially observable markov decision processes.  
I focused on environments that require good use of context, but **not** very long or variable-length context. Because of this constraint grid based problems seemed like a good fit because a lot can unfold in the environment in relatively few time-steps.

The results show that transformers can be trained to use context effectively in a reasonable amount of time on consumer hardware with on-policy reinforcement learning.

---

## Approach

The key idea is to treat **each trajectory** as the atomic unit of learning.  
I treat each time step as one "token" in the transformer and I consider the last **last action** taken and **last reward** received to be part of the observation (shouldn't the agents always be able to remember what action they took?)  

Because the entire context influences action selection, training on only part of a trajectory would make the data off-policy.
This makes this approach somewhat similar to monte carlo learning but you can still use GAE to control variance.

A rollout is collected over multiple parallel agents, using a KV cache to speed up inference.

When the rollout is filled and the episodes are finished
* The advantage and target values are calculated using **Generalized Advantage Estimation (GAE)**. 
* The trajectories are shuffled over the batch dimension but not the timestep dimension.
* The trajectories are split into multiple mini-batches over the batch dimension.


The mini-batch update uses a standard PPO loss function and the adamw or muon optimizer to make a gradient step.

---

## Architecture

### Embeddings
The model uses three embeddings:
- **Last action** — similar to a tied-weight embedding in an LLM, also reused as the actor head
- **Last reward** — a simple linear projection
- **Observation** — environment-specific; for grid environments, a small CNN

These embeddings are summed and passed into a stack of transformer layers.

### Transformer Layers
- Pre-norm
- RoPE positional embeddings
- Grouped-query attention
- Optional sliding-window attention

### Outputs
- **Actor head** — transposed action embedding
- **Value function** — includes an extra hidden layer + activation (slightly better than a direct linear projection)

I've been testing with 6 transformer layers, a hidden size of 128 and a feed forward size of 768 with a context size of 512.
The total parameter count comes out to around 2.3 million parameters.

---

## Performance

One thing I wondered about was if you could get the number of samples needed to train with PPO using a transformer based model on consumer hardware and I found that it can fairly fast with a small context size!

I'm training on a single 5090 at **2 million steps** per second!

Here are a few things that were important to performance

* Both the environments and training code where written in a single end-to-end jitted JAX training loop.
* Using the cudnn backend on nvidia GPU's via [Dot Product Attention](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html)
* bfloat16 with float32 accumulation speeds up training and didn't noticeably hurt performance.
* Using grouped query attention with one kv head and four query heads significantly speeds up training and has a small negative impact on performance.
*Batched inference has a large impact on performance, using 4092 vectorized agents insured that rollout creation had high algorithmic intensity.


---

## Results


To test memory and context usage, I used a simple 2D grid environment:
- A new grid is generated each episode using multiple octaves of perlin noise.
- The agent must find a random target location.
- Upon reaching the target, the agent gets a reward and is moved to another random location.
- Maximizing rewards requires **exploring effectively** and **remembering** the route back using features of the grid.

Observations are given as a small grid view of one-hot tile types, encoded with a small CNN before being passed to the transformer.


