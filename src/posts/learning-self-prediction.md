

Both model weights and context are a form of state that adapts to experience, one through the RNN or transformer and one through the gradient step and RL algorithm. One key distinction is the RL algorithem can learn to seek state that is useful for the sequence model but not state that is useful for it's gradient step or another agents gradient step.


Over the trajectory of the sequence model/episode actions that gather information that can be used to make better decisions latter is the "greedy" action.

Over the entire training session, actions that lead to a better policy would be "greedy" over the cumulative reward off all episodes if the model is capable of predicting what actions will lead to experience that leads to better decisions.

Sequence state like a KV cache is not fundamentally different in theory than parameter state.


Exploration is connected to communication. Exploritory actions are to change your own beilifs or state and communication is to change anothers belifs or state.