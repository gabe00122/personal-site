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

<script>
    import VideoPlayer from "../routes/components/video.svelte";
</script>

# Introduction

* Part 1 (this post) - Basic implementation
* Part 2 - Vectorized training
* Part 3 - Adam and regularization
* Part 4 - Self play for learning muti-agent environments
* Part 5 (bonus) - Deploying the model to the web with tensorflow.js

I recently finished reading Reinforcement Learning: An Introduction by Sutton and Barto. I decided to implement the actor critic algorithm outlined in the book and learned a lot in the process.

## Algorithm Gist
An actor critic algorithm is a machine learning algorithm that use a critic model to teach a actor model to take actions that will maximize some future reward.  These models can use any sort of functional approximation but in this tutorial we use artificial neural networks.

If you have a function to approximate the cumulative reward over an entire episode and you take the difference for the reward prediction with observations from the `n` and `n+1` time steps that gives you a expected instantaneous reward at the `n` time step. The difference between the expected reward and the received reward is called temporal difference error. Temporal difference error can be minimized with semi-gradient decent to improve our critic predictions but it can also be used to make a action preference in the policy stronger or weaker depending on if the error is positive or negative.  

Action selection driven by the temporal difference error will converge towards a preference that matches how likely it is that a given action will yield the highest cumulative reward.


Here is the complete pseudo code for a one step actor critic, note this is not the only kind of actor-critic for example you could use n-steps or eligibility traces or calculated the average reward instead of the cumulative reward.

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
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}}I\gamma\nabla \ln \pi(A|S,\boldsymbol{\theta})$  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$I \leftarrow \gamma I$  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$S \leftarrow S'$

## Implementation

### Network architecture
Since we're using deep learning for our functional approximation first we need to define a our network architecture with flax.  
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

### Defining some types

At this point we could initialize a network and use it as our agent but without the right parameters it likely won't make very good decisions. We need a way to optimize the parameters, so let's start by defining some types to help organize.
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
Using `static_argnums` tells jax to treat the self object as a static parameter for jit compilation. Calling this function with a observation vector from our environment will give us an action from the current policy.
```python
    @partial(jax.jit, static_argnums=(0,))
    def sample_action(
        self, training_state: TrainingState, obs: ArrayLike, rng_key: ArrayLike
    ):
        logits = self.actor_model.apply(training_state.actor_params, obs)
        return random.categorical(rng_key, logits)
```

#### Updating the model
Now were to the real meat and potato's of the implementation. Here we update the model weights using the n observation the n+1 observation and the reward in-between.  
First we gather our hyper parameters (note these do not need to be on a schedule, this is just personal preference)
```python
def update_models(
    self, training_state: TrainingState, params: ModelUpdateParams
) -> tuple[TrainingState, Metrics]:
    actor_params = training_state.actor_params
    critic_params = training_state.critic_params
    importance = training_state.importance
    step = params.step

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
```

Now we can calculate our TD error, this is the difference between receded reward and expected reward.  
This corresponds to $\delta \leftarrow R + \gamma \hat{\upsilon}(S', \bold{w}) - \hat{\upsilon}(S, \bold{w})$  
```python
def update_models(
    self, training_state: TrainingState, params: ModelUpdateParams
) -> tuple[TrainingState, Metrics]:
    ...
    
    state_value = self.critic_model.apply(critic_params, obs)
    td_error = rewards - state_value
    td_error += jax.lax.cond(
        done,
        lambda: 0.0,  # if the episode is over our next predicted reward is always zero
        lambda: discount * self.critic_model.apply(critic_params, next_obs)
    )
```

Updating the critic  
This corresponds to $\bold{w} \leftarrow \bold{w} + \alpha^\bold{w} \delta \nabla \hat\upsilon (S,\bold{w})$  
```python
def update_params(params, grad, step_size):
    return jax.tree_map(
        lambda param, grad_param: param + step_size * grad_param,
        params, grad
    )


def update_models(
    self, training_state: TrainingState, params: ModelUpdateParams
) -> tuple[TrainingState, Metrics]:
    ...

    critic_gradient = jax.grad(self.critic_model.apply)(critic_params, obs)
    critic_params = update_params(
        critic_params,
        critic_gradient,
        critic_learning_rate * td_error,
    )
```

And finally updating the actor  
This corresponds to $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}}I\gamma\nabla \ln \pi(A|S,\boldsymbol{\theta})$  
```python
def action_log_probability(self, actor_params, obs: ArrayLike, action: ArrayLike):
    logits = self.actor_model.apply(actor_params, obs)
    return nn.log_softmax(logits)[action]

# Update the actor
def update_models(
    self, training_state: TrainingState, params: ModelUpdateParams
) -> tuple[TrainingState, Metrics]:
    ...

    actor_gradient = jax.grad(self.action_log_probability)(actor_params, obs, actions)
    actor_params = update_params(
        actor_params,
        actor_gradient,
        actor_learning_rate * importance * td_error,
    )
```

Training loop with [gym](https://gymnasium.farama.org/)
```python
for step in range(total_steps):
    rng_key, action_key = random.split(rng_key)
    action = actor_critic.sample_action(training_state, obs, action_key)

    next_obs, reward, terminated, truncated, _ = env.step(action.item())
    done = terminated or truncated

    model_update_params = ModelUpdateParams(
        step=step,
        obs=obs,
        actions=action,
        rewards=reward,
        next_obs=next_obs,
        done=done,
    )

    training_state, metrics = actor_critic.update_models(training_state, model_update_params)
    obs = next_obs

    if done:
        obs, _ = env.reset()
```

For a full example of the code see: https://github.com/gabe00122/tutorial_actor_critic/tree/main/tutorial_actor_critic/part1

Training results:

![image](/blog/actorcritic/cartpole-rewards.png)

<VideoPlayer />