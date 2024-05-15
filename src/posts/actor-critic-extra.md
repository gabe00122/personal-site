## Design Choice 1 - Critic Type
1. Estimate the discounted cumulative reward
	* Advantages: It's easy to tell if your critic values are accurate
	* Disadvantages: Episodes with many steps lead can lead to cumulative rewards that are very high. Not equiped to deal for very long time episodes.
2. Estimate the rate of reward
	* Advantages: Better theoretical way to handle long episodes.

In this implementation we use cumulative rewards.

## Design Choice 2 - Number of steps
1. single-step
    * TD fixed point is closer to the true local minima for the policy
2. n-step
    * TD fixed point is farther away with more steps but the algorithm can converge faster
    * More steps can be more computationally costly
3. eligibility traces
    * Similar to n-step TD but represented as a trace vector of weights
    * Can represent a large number of steps with a fixed computational cost

In this implementation we use single step because it's easier to implement
