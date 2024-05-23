## Design Choice 1 - Critic Type

1. Estimate the discounted cumulative reward
   - Advantages: It's easy to tell if your critic values are accurate
   - Disadvantages: Episodes with many steps lead can lead to cumulative rewards that are very high. Not equiped to deal for very long time episodes.
2. Estimate the rate of reward
   - Advantages: Better theoretical way to handle long episodes.

In this implementation we use cumulative rewards.

## Design Choice 2 - Number of steps

1. single-step
   - TD fixed point is closer to the true local minima for the policy
2. n-step
   - TD fixed point is farther away with more steps but the algorithm can converge faster
   - More steps can be more computationally costly
3. eligibility traces
   - Similar to n-step TD but represented as a trace vector of weights
   - Can represent a large number of steps with a fixed computational cost

In this implementation we use single step because it's easier to implement

### Our actor critic class

We can define a class in jax to help us organize the algorithm and make it easier to swap it out with other RL algorithms later.

```python
from optax import Schedule
from dataclasses import dataclass

class TrainingState(NamedTuple):
    importance: ArrayLike # a float scaler and I from the algorithm
    actor_params: Any     # θ parameters from our actor model
    critic_params: Any    # w parameters from our critic model


class HyperParameters(NamedTuple):
    discount: Schedule             # this is γ
    actor_learning_rate: Schedule  # this is αθ
    critic_learning_rate: Schedule # this is αw


class ModelUpdateParams(NamedTuple):
    step: ArrayLike     # total training steps used for hyper parameter schedule

    obs: ArrayLike      # this is S a floating point vector
    action: ArrayLike   # this is A a integer scaler representing which discrete action was selected
    reward: ArrayLike   # this is R a floating point scaler
    next_obs: ArrayLike # this is S′ a floating point vector representing the observation following taking the action
    done: ArrayLike     # this is a boolean scaler for if S′ is a terminal state

@dataclass(frozen=True)
class ActorCritic:
    actor_model: nn.Module  # π model with θ parameters
    critic_model: nn.Module # Υ model with w parameters
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
