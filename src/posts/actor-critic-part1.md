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

## Design Choice 1 - Critic Type
1. Estimate the discounted culmlative reward
	* Advantages: It's easy to tell if your critic values are accurate
	* Disadvantages: Episodes with many steps lead can lead to culmlative rewards that are very high. Not equiped to deal for very long time episodes.
2. Estimate the rate of reward
	* Advantages: Better theoretical way to handle long episodes.

In this implementation we use cumulative rewards.

## Design Choice 2 - Number of steps
1. single-step
    * TD fixed point is closer to the true local minima for the policy
2. n-step
    * TD fixed point is farther away with more steps but the algorithem can converge faster
    * More steps can be more computationly costly
3. eligibility traces
    * Similar to n-step TD but represented as a trace vector of weights
    * Can represent a large number of steps with a fixed computational cost

In this implementation we use single step because it's easier to implement


## One-step Actor-Critic (episodic), for estimating $\pi_\theta\approx\pi_*$
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

## Implementation

### Network Arcetecture
Since we're using deep learning for our functional approximation first we need to define a our network arcetecture with flax.  
See: https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html#module-basics for more details
```python
from flax import linen as nn
from collections.abc import Callable, Sequence

class MlpBody(nn.Module):
    features: Sequence[int]
    activation: Callable[[Array], Array] = nn.relu

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        for i, feat in enumerate(self.features):
            x = nn.Dense(
                feat,
                name=f"mlp_layer_{i}",
                kernel_init=nn.initializers.he_normal(),
            )(x)
            x = self.activation(x)
        return x

# The critic head outputs a scaler value which can be used as a value function
class CriticHead(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        value = nn.Dense(
            1,
            name="critic_head",
            kernel_init=nn.initializers.he_uniform(),
        )(inputs)
        return jnp.squeeze(value)

# This descrete actor head outputs a logit for each of n-number possible actions, these logits will be used to represent a distribution of actor preferences.
# The goal of our algorithem is to increase the preference for good actions and descrease the prefence for bad actions.
class ActorHead(nn.Module):
    actions: int

    @nn.compact
    def __call__(self, inputs):
        actor_logits = nn.Dense(
            self.actions,
            name="actor_head",
            kernel_init=nn.initializers.he_uniform(),
        )(inputs)

        return actor_logits

# We can define full networks by sequencing the mlp with the actor and critic heads
actor_model = nn.Sequential((
    MlpBody(features=(64, 64)),
    ActorHead(actions=action_space),
))
critic_model = nn.Sequential((
    MlpBody(features=(64, 64)),
    CriticHead(),
))
```

### Actor Critic class

At this point we could initalize a network and use it as our agent but without the right parameters it likely won't make very good decisions. We need a way to optimize the parameters, so let's start by defining some types to help orginize.
Optax provides an easy way to define hyperparameter schedules.
```python
from optax import Schedule
from dataclasses import dataclass

class TrainingState(NamedTuple):
    importance: ArrayLike # a float scaler and I from the algorithem
    actor_params: Any     # parameters from our actor network
    critic_params: Any    # parameters from our critic networl


class Metrics(NamedTuple):
    td_error: float
    state_value: float


class HyperParameters(NamedTuple):
    discount: Schedule             # this is γ
    actor_learning_rate: Schedule  # this is αw
    critic_learning_rate: Schedule # this is αθ


class ModelUpdateParams(NamedTuple):
    step: ArrayLike     # total training steps used for hyper parameter schedule

    obs: ArrayLike      # this is S
    actions: ArrayLike  # this is A
    rewards: ArrayLike  # this is R
    next_obs: ArrayLike # this is S′
    done: ArrayLike     # this is a boolean for if S′ is a terminal state

@dataclass(frozen=True)
class ActorCritic:
    actor_model: nn.Module  # w
    critic_model: nn.Module # θ
    hyper_parameters: HyperParameters

    def init(self, state_space: int, rng_key: ArrayLike) -> TrainingState:
        ...
    
    def sample_action(
        self, training_state: TrainingState, obs: ArrayLike, rng_key: ArrayLike
    ):
        ...

    def action_log_probability(self, actor_params, obs: ArrayLike, action: ArrayLike):
        ...
    
    def update_models(
        self, training_state: TrainingState, params: ModelUpdateParams
    ) -> tuple[TrainingState, Metrics]:
        ...
```

#### Sample Action
We need to define the sample action function which represents $\pi (\sdot|S,\boldsymbol{\theta})$ from our formula above.  
Luckily jax can already sample from a logit distribution with `random.categorical`  
Using `static_argnums` tells jax to treat the self object as a static parameter for jit compilation.  
```python
    @partial(jax.jit, static_argnums=(0,))
    def sample_action(
        self, training_state: TrainingState, obs: ArrayLike, rng_key: ArrayLike
    ):
        logits = self.actor_model.apply(training_state.actor_params, obs)
        return random.categorical(rng_key, logits)
```