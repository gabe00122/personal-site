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

I recently finished reading Reinforcement Learning: An Introduction by Sutton and Barto and thought it would be a good learning experience to implement some of the algorithems in the book myself. The goal of these blog posts are to document my experiements implmenting an actor critic based on that book and beyond. I used this actor critic implementation first learn relativly simple problems like balancing a pole on a cart or swinging a penulum, next i used to to achieve decent results on the gym enviroment lunar lander. Finally I used this algorithem to learn to play tic tac toe as well as a softmax algorithem purly from experience playing against itself.


Namely they provide some mechnism to handle the exploration vs explitoation trade off that seems better then the near greedy popular in Q learning based methods. This means that the probability that an agent takes an action should roughly match the odds that the actions provides the best rewards (this is known as the matching law in physcology).

Furthermoor they can be applied to both continusous or decrete actions spaces giving these algorithems a wide range of applicability.

To implement these methods I choose to use jax because it has the potential to provide very fast RL training if the enviroment is also written with jax.

* Part 1 (this post) - Basic implementation
* Part 2 - Vectorized training
* Part 3 - Adam and regularization
* Part 4 - Self play for learning muti-agent enviroements
* Part 5 (bonus) - Deploying the model to the web with tensorflow.js

### Algorithem Overview

An actor critic is made up of two primary components.

First the actor which:
When given a observation of an enviroment predicts an action that will maximize a reward signal for the agent.

And a critic which:
When given a observation of an enviroment predicts the expected reward that will be received (either cumulativly or as an rate of reward over time)


The purpose of the ciritc is to calculated temproal difference error is a measure of the change in reward expectation over one or more enviroment steps. 
Calculated it over a single step is called TD(0) which is what I implemented.

The definition of a 1-step actor critic is as follows:

Input: a dierentiable policy parameterization ⇡(a|s,✓)

Input: a di↵erentiable state-value function parameterization ˆ v(s,w)

Parameters: step sizes ↵✓ > 0, ↵w > 0 

Initialize policy parameter ✓ 2 Rd0 and state-value weights w 2 Rd (e.g., to 0) Loop forever (for each episode): Initialize S (first state of episode) I 1 Loop while S is not terminal (for each time step): A⇠⇡(·|S,✓) Take action A, observe S0,R R+ ˆ v(S0,w) ˆ v(S,w) ( if S0 is terminal, then ˆ v(S0,w) . = 0) w w+↵wrˆ v(S,w) ✓ ✓+↵✓I rln⇡(A|S,✓) I I S S0