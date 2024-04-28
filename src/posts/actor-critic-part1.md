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

# Introdution

* Part 1 (this post) - Basic implementation
* Part 2 - Vectorized training
* Part 3 - Adam and regularization
* Part 4 - Self play for learning muti-agent enviroements
* Part 5 (bonus) - Deploying the model to the web with tensorflow.js

I recently finished reading Reinforcement Learning: An Introduction by Sutton and Barto. I decided to implement the actor critic algorithm outlined in the book and learned a lot in the process.

In the series I'll outline how I implemented an actor critic and give some pointers on some of the different design choices for this algorithm.

## Algorithm Gist
An actor critic algorithm is a machine learning algorithm that use a critic model to teach a actor model to take actions that will maximize some future reward.  These models can use any sort of functional approximation but in this tutorial we use artificial neural networks.

## Why are they cool?
* They learn to match they're action preferences to the probablity that a action will lead to the best reward (see https://en.wikipedia.org/wiki/Matching_law)
* They can be used for both discrete actions and contiousus actions

### Design Choice 1 - Critic Type
1. Estimate the discounted culmlative reward
	* Advantages: It's easy to tell if your critic values are accurate
	* Disadvantages: Episodes with many steps lead can lead to culmlative rewards that are very high. Not equiped to deal for very long time episodes.
2. Estimate the rate of reward
	* Advantages: Better theoretical way to handle long episodes.

In this implementation we use cumulative rewards.

### Design Choice 2 - Number of steps
1. single-step
2. n-step
3. Eligibility traces

In this implementation we use single step because it's easier to implement


#### One-step Actor-Critic (episodic), for estimating $\pi_\theta\approx\pi_*$
> Input: a differentiable policy parameterization $\pi(a|s,\boldsymbol{\theta})$  
> Input: a differentiable state-value function parameterization $\hat{\upsilon}(s,\bold{w})$  
> Parameters: step sizes $\alpha^\theta > 0, \alpha^w > 0$  
> Initialize policy parameter $\bold{\theta}\in\R^{d'}$ and state-value weights $\bold{w}\in\R^d$ (e.g., to 0)  
> Loop forever (for each episode):  
> &nbsp;&nbsp;&nbsp;&nbsp;Initialize $S$ (first state of the episode):  
> &nbsp;&nbsp;&nbsp;&nbsp;$\textit{I} \leftarrow 1$  
> &nbsp;&nbsp;&nbsp;&nbsp;Loop while $S$ is not terminal (for each time step):  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$A \backsim \pi (\sdot|S,\boldsymbol{\theta})$  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Take action $A$, observe $S'$, $R$  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\delta \leftarrow R + \gamma \hat{\upsilon}(S', \bold{w}) - \hat{\upsilon}(S, \bold{w})$  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\bold{w} \leftarrow \bold{w} + \alpha^\bold{w} \delta \nabla \hat\upsilon (S,\bold{w})$  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}}I\nabla \ln \pi(A|S,\boldsymbol{\theta})$  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$I \leftarrow \gamma I$  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$S \leftarrow S'$

#### Implementation


```python
import jax
from jax import random, numpy as jnp
from jax.typing import ArrayLike
from flax import linen as nn
from optax import Schedule
from typing import NamedTuple, Any


class TrainingState(NamedTuple):
    actor_params: Any
    critic_params: Any


class Metrics(NamedTuple):
    td_error: float
    state_value: float


class HyperParameters(NamedTuple):
    discount: Schedule
    actor_learning_rate: Schedule
    critic_learning_rate: Schedule


class ModelUpdateParams(NamedTuple):
    step: ArrayLike
    importance: ArrayLike

    # Just remember SARS(A)
    obs: ArrayLike  # S
    actions: ArrayLike  # A
    rewards: ArrayLike  # R
    next_obs: ArrayLike  # S
    done: ArrayLike


class ActorCritic:
    def __init__(
        self,
        actor_model: nn.Module,
        critic_model: nn.Module,
        hyper_parameters: HyperParameters,
    ):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.hyper_parameters = hyper_parameters

    def sample_action(
        self, training_state: TrainingState, obs: ArrayLike, key: ArrayLike
    ):
        logits = self.actor_model.apply(training_state.actor_params, obs)
        return random.categorical(key, logits)

    def action_log_probability(self, actor_params, obs: ArrayLike, action: ArrayLike):
        logits = self.actor_model.apply(actor_params, obs)
        action_softmax = nn.softmax(logits)
        selected_action_softmax = action_softmax[action]

        return jnp.log(selected_action_softmax)

    def update_models(
        self, training_state: TrainingState, params: ModelUpdateParams
    ) -> tuple[TrainingState, Metrics]:
        actor_params = training_state.actor_params
        critic_params = training_state.critic_params
        step = params.step
        importance = params.importance

        # Just remember SARSA expect we skip the final action here
        obs = params.obs  # State
        actions = params.actions  # Action
        rewards = params.rewards  # Reward
        next_obs = params.next_obs  # State
        done = params.done  # Also State

        # Let's calculate are hyperparameters from the schedule
        discount = self.hyper_parameters.discount(step)
        actor_learning_rate = self.hyper_parameters.actor_learning_rate(step)
        critic_learning_rate = self.hyper_parameters.critic_learning_rate(step)

        # Calculate the TD error
        state_value = self.critic_model.apply(critic_params, obs)
        td_error = rewards - state_value
        if not done:
            td_error += discount * self.critic_model.apply(critic_params, next_obs)

        # Update the critic
        critic_gradient = jax.grad(self.critic_model.apply)(critic_params, obs)
        critic_params = update_params(
            critic_params,
            critic_gradient,
            critic_learning_rate * td_error,
        )

        # Update the actor
        actor_gradient = jax.grad(self.action_log_probability)(
            actor_params, obs, actions
        )
        actor_params = update_params(
            actor_params,
            actor_gradient,
            actor_learning_rate * importance * td_error,
        )

        # Record Metrics
        metrics = Metrics(td_error, state_value)

        return TrainingState(actor_params, critic_params), metrics


def update_params(params, grad, step_size):
    return jax.tree_map(
        lambda param, grad_param: params + step_size * grad_param, params, grad
    )

```