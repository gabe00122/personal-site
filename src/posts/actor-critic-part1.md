---
title: Basic Actor Critic
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

I recently finished reading [Reinforcement Learning: An Introduction by Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html). Inspired by the book, I decided to implement the actor-critic algorithm and learned a lot in the process. My first step was translating the pseudocode outlined in the book into JAX. If you're new to reinforcement learning, it's better to start by implementing a solution to a multi-armed bandit problem and familiarize yourself with temporal difference learning.

## Reinforcement Learning

Reinforcement learning is a class of algorithms designed to train a model, called a policy, to take actions that maximize the total reward received over time. It is called reinforcement learning because rewards reinforce behaviors that lead to higher rewards.

These algorithms require functional approximation, with deep learning being a particularly promising approach.

## Actor Critics

Actor-critic algorithms are a class of reinforcement learning algorithms that use two models to teach each other: an actor and a critic. Both the actor and the critic take an observation of the environment as input but produce different outputs.

- The actor outputs an action distribution, from which an action can be sampled.
- The critic outputs a prediction of the total reward remaining in the episode or a prediction of the future rate of reward.

The critic's predictions always depend on the current policy; a better policy generally leads to more rewards. Thus, when the actor improves, the critic needs to be updated to match the improved performance of the new actor.

The actor model can use the critic's value estimations to adjust the action distribution towards actions that lead to better reward predictions.

In this way, the actor and critic engage in a balancing act, continually adjusting towards a better policy.

# Implementation

I started with the pseudocode for a one-step actor critic from Sutton and Barto, I highly recommend reading this book yourself but I'll do my best translating the notation into a plain english explanation the best I can.

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

TODO: Cite page number

Some notes on the above:

- Variables
  - $S$ is the observation from the environment or "state" this is the input both the actor and critic get about the environment during the current time step
  - $\gamma$ is the discount, this is used to value rewards more farther in the future less the sooner rewards
  - $I$ stands for the importance, it starts about at 1.0 but diminish's with the discount over the course of the episode. We use it to scale the actor learning rate down the more the episode has progressed.
- Functions
  - $\pi (\sdot|S,\boldsymbol{\theta})$ this is a function that samples a random action out of the action distribution
  - $\pi(A|S,\boldsymbol{\theta})$ this gives of the likelihood a given action would be selected under the current policy
  - $\hat{\upsilon}(S, \bold{w})$ this is the estimate of the cumulative discounted reward starting at S

## Creating Deep Neural Networks

I'm not going to go into great depth on this since there are plenty of other resources, but you can see how I initialized my neural networks for this algorithm here: [link](https://github.com/gabe00122/tutorial_actor_critic/blob/main/tutorial_actor_critic/mlp.py)  
Also see the flax docs on how to initialize networks in jax: [link](https://flax.readthedocs.io/en/latest/quick_start.html#define-network)

## Sample an action

$\pi (\sdot|S,\boldsymbol{\theta})$

We need a way to get select actions from our policy, the solution to this depends on what type of actions are required, for example continuous or discrete.
For this tutorial we assume our action will always be one from a set of discrete actions, i.e A, B, C, or D
A common way to handle this is to interpret the outputs of the actor model as [logits](https://en.wikipedia.org/wiki/Logit) for each of the possible actions. Lucky numpy (and therefor jax) has a built in function for picking a random action from a set of logits.

```python
def sample_action(
    actor_model, training_state: TrainingState, obs: ArrayLike, rng_key: ArrayLike
):
    logits = self.actor_model.apply(training_state.actor_params, obs)
    return random.categorical(rng_key, logits)
```

## Update our parameters

### Calculating the TD error

$\delta \leftarrow R + \gamma \hat{\upsilon}(S', \bold{w}) - \hat{\upsilon}(S, \bold{w})$

If our value function $\hat{\upsilon}(S, \bold{w})$ is an estimation of the total reward left to receive at observation S then. Then $\hat{\upsilon}(S, \bold{w}) - \hat{\upsilon}(S+n, \bold{w})$ should be an estimation of the reward received between S and S+n. If n is 1 meaning it's the next observation then this difference should be an estimation of the instantaneous reward at S. The difference between the estimated reward and the actual instantaneous reward is the temporal difference error $\delta$.

If the observation is the episode terminal (done = True) than no further rewards are possible and so the state value should always be 0.0

This TD error will be used to update both our critic to make better estimates and our actor to take actions that lead to more rewards.

```python
def temporal_difference_error(
    critic_model, critic_params, discount, observation, reward, next_observation, done
) -> float:
    state_value = critic_model.apply(critic_params, observation)
    next_state_value = jax.lax.cond(
        done,
        lambda: 0.0,
        lambda: discount * critic_model.apply(critic_params, next_observation)
    )

    estimated_reward = state_value - next_state_value
    td_error = rewards - estimated_reward

    return td_error
```

### Updating the critic

$\bold{w} \leftarrow \bold{w} + \alpha^\bold{w} \delta \nabla \hat\upsilon (S,\bold{w})$

The temporal difference error can be used to improve the critic predictions by moving in the direction of the gradient at S times the td_error.
Value(S+1) + reward is a better prediction than value(S) and our td error already represents this difference.
If we simply move a step in the direction of the gradient of the value at S multiplied by the td error our value estimations will improve.
We're also providing a way for our later state predictions to influence our earlier state predictions
so that with enough repetitions the predictions at the end of the episode should help determine the predictions at the beginning of the episode.

This is technically semi-gradient decent because the gradient at value(S+1) is ignored,
a consequence of this is that the value function doesn't converge to the true reward approximation but a point close to it called td fixed point.

```python
def update_params(params, grad, step_size):
    return jax.tree_map(
        lambda param, grad_param: param + step_size * grad_param,
        params, grad
    )


def update_critic(
    critic_model, critic_params, obs, critic_learning_rate, td_error
):
    critic_gradient = jax.grad(critic_model.apply)(critic_params, obs)
    critic_params = update_params(
        critic_params,
        critic_gradient,
        critic_learning_rate * td_error,
    )

    return critic_params
```

### Updating the actor

$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}}I\gamma\nabla \ln \pi(A|S,\boldsymbol{\theta})$

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

A line plot showing the sum reward for each episode, averaged over twenty separate training seeds
![A line plot showing the sum reward for each episode, averaged over twenty separate training seeds](/blog/actorcritic/cartpole-rewards.png)

<VideoPlayer url="/blog/actorcritic/cartpole-post-training.mp4" />
