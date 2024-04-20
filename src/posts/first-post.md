---
title: Actor Critic (part 1)
description: Implement a actor critic from scratch in jax
date: '2023-4-14'
categories:
  - reinforcement learning
  - jax
  - flax
  - actor critic
  - deep learning
published: true
---

## Introduction

In this series I'm going to talk about how I implemented a custom actor critic algorithem in jax. This project was primally meant as a learning project. If your just trying to get results on an enviroment it would be better to use something like PPO but I think the exercise of a custom implementation is still intresting.

* Part 1 (this post) - Basic implementation
* Part 2 - Vectorized training
* Part 3 - Adam and regularization
* Part 4 - Self play for learning muti-agent enviroements
* Part 5 (bonus) - Deploying the model to the web with tensorflow.js

### Algorithem Overview

An actor critic algorithem uses temporal difference learning to predict the expected reward, this can either be a cumulative reward for the entire episode or a rate of reward per enviroment step. While for episodic enviroments discounted rewards work, these have theoretical issues for continousus enviroements and a rate of reward should be used instead. Never the less it is more common to implemented it using a discounted cumulative reward and for this implementation that's what we use.

The definition of a 1-step actor critic is as follows:

Input: a dierentiable policy parameterization ⇡(a|s,✓)

Input: a di↵erentiable state-value function parameterization ˆ v(s,w)

Parameters: step sizes ↵✓ > 0, ↵w > 0 

Initialize policy parameter ✓ 2 Rd0 and state-value weights w 2 Rd (e.g., to 0) Loop forever (for each episode): Initialize S (first state of episode) I 1 Loop while S is not terminal (for each time step): A⇠⇡(·|S,✓) Take action A, observe S0,R R+ ˆ v(S0,w) ˆ v(S,w) ( if S0 is terminal, then ˆ v(S0,w) . = 0) w w+↵wrˆ v(S,w) ✓ ✓+↵✓I rln⇡(A|S,✓) I I S S0