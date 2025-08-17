
When certain information is in the sequence model it can later make better decisions. If the model knows it has information that will let it make better decisions the value function goes up so actions that lead to that information will be reinforced. This is a learned exploration strategy. When a models parameters are updated with certain key facts the performance goes up, aka the cumulative reward over the entire training run rather than just the episode but there is no outer value function that is aware the model just improved so information that leads to a better model is not sought out my the policy. I believe information that is valuable to a sequence model can be seen as a kind of meta learned exploration and is not fundamentally different than information that is beneficial to making gradient steps. In context learning is not fundamentally different that other kinds of learning. The key difference is that with a sequence model there's an outer loop that can learn what kinds of information the inner loop should seek.


Because any state change that increases the probability of reward should increase the advantage estimate with repeated trials.
[11:48 PM]gabe00122: The critic can learn the sorts of sequence model information that will be useful later in terms of increased expected reward
[11:49 PM]gabe00122: So actions that lead to adding that information to the sequence model should also have an advantage.
[11:49 PM]gabe00122: Like my grid agents are generally pretty good at exploring the grid.
[11:50 PM]gabe00122: Not perfect but they learned not the explore the same spot twice for the most part.
[11:51 PM]gabe00122: It's because the advantage for exploring the same place again is lower than going there the first time
[11:51 PM]gabe00122: Because a memory of the reward not being there is in the seqence model already
[11:52 PM]gabe00122: You often don't know for certain information will be useful but you can make a statistical guess.
[11:54 PM]gabe00122: The big issue imo is this only helps with a sequence model. It does not lead to what helps the model parameters get better.
[11:54 PM]gabe00122: Unless those things happen to align, but I think they do not in some importent cases.
[11:55 PM]gabe00122: If we could have a model that would seek information that could improve it's parameters that would be powerful
[11:56 PM]gabe00122: A bad outer loop could make a good in context learner but the outer loop is still bad


Exploration is 