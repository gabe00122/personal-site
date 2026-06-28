---

title: Value functions for LLMs
description: Value functions for LLM fine-tuning
date: '2026-06-12'
published: true
---------------

<script>
    import Image from "$lib/components/image.svelte";
    import EpisodeViewer from "$lib/components/episode/episodeViewer.svelte";
    import ArchitectureDiagram from "$lib/components/valmArchitecture.svelte";
</script>

# Introduction

Some of the earliest applications of RL to LLMs with RLHF were based on PPO, with InstructGPT being explicitly PPO-based. Since then, GRPO has removed the critic and popularized LLM RL reasoning training at scale, becoming the dominant baseline. More recently, the literature has split between a "critic-free is enough" camp (GRPO, RLOO, GSPO, and REINFORCE++) and a "bring the critic back" camp (VC-PPO, VAPO, AsyPPO mini-critics, and GenAC).

The thesis of this post is that early LLM PPO implementations were limited by the quality of their value estimates, but value approximation itself remains a promising technique if the architecture can make value prediction cheap and accurate.

GRPO has the advantage of not requiring a critic, but it can only generate a sequence-level baseline, and it does so at the cost of sacrificing prompt diversity. Despite these advantages GRPO struggles in comparison to PPO on classis RL environments (https://arxiv.org/pdf/2511.03527). By contrast, a good value function doesn't just learn the expected return for an entire sequence; it is a learned heuristic for every partial sequence. In principle, that heuristic can use information from the model’s own representation of the state, learn from a single rollout without a group, and utilize information from state transitions through bootstrapping.

My experiment fine-tunes Qwen3 on Wordle using a small value model that consumes representations from the base model. This architecture provides a critic that adds little overhead compared with the policy update while providing a more granular baseline than sequence-level group normalization. Contrarly to other works bootstraping is shown to outpreform monte carlo returns with this critic arcetecture. I'm able to train Qwen3-4B-Instruct-2507 from a 1% solve rate on Wordle to 99% in 11 hours on a RTX 5090 GPU.

# Why critics fell out of favor

Historically, learned value functions for LLM PPO have usually taken one of two forms. The first is to train a value model at roughly LLM scale, which is expensive. The second is to attach a small value head to the final hidden state before the language-model projection. This is much cheaper, but it creates a different problem: the final hidden state is tightly coupled to the next-token distribution. Because almost every direction in the final latent affects the logits, attaching a value loss there risks fighting the policy objective.

# Architecture

A growing body of evidence suggests that latents from deeper layers of the base model provide better representations for downstream predictions. PING (https://www.medrxiv.org/content/10.1101/2025.09.17.25336018v1) My approach is to take every n-th latent and map it to a smaller value prediction model.

Training an existing policy with a randomly initialized value network could be harmful to the policy (VC-PPO), since the value may be very far from the true mean return. One solution is to warm up the value function offline on a frozen policy before online learning. However, with a frozen base model, this gives the value network no way to integrate history and context in a way that differs from the base models attentional patterns. Being able to learn a function of history is particularly important for value prediction because it is a long-term forecast of the future and is highly dependent on hidden environment state.

This is why I designed the value network as a separate, parallel, ladder-style transformer that consumes latents. The attention patterns for the value network can be learned during the offline critic warmup without affecting the base model's attention.

This is effectively ladder side tuning with respect to the value function and lora training with respect to the policy.

<ArchitectureDiagram />

The two streams are:

**Base layer** — a Qwen3 layer with a LoRA adapter. Stacked together these form the policy, with the residual stream running from the token embedding up to the token prediction.

**Value layer** — a scaled-down version of the Qwen3 architecture with its own attention layers. These form the parallel value network, running from the last reward up to the value prediction.

**Value encode** — every n-th base latent is projected into the value stream through a stop-gradient SwiGLU block, so gradients from the value loss never flow back into the base model:

```python
x = jax.lax.stop_gradient(x)
x = self._normalize(x)
gate = self._up_gate(x)
x = self._encode_up(x)
x = jax.nn.silu(x) * gate
x = self._encode_down(x)
```

I use a swiglu style mlp block with normalization so the value network can selectively filter latents from the base model into it's residual stream.

# Environment

Although it could be applied to different tasks I primarily tested this method on wordle. Wordle here is defined as a pomdp and rather then a contextual bandit. A transition is a single token, not a turn. An episode is multiple turns (up to 6 guesses) capped at 1024 tokens.

## Reward

The reward function is split into two components, bith designed so the maximum return is 1.0.

Partial credit, granted once per slot, the first time a slot is revealed:
* +0.025 the first time slot i is yellow-or-green
* +0.025 the first time slot i is green

Each of the 5 slots can contribute at most 0.05, so partial credit tops out at 0.25.

Terminal bonus: +0.75 when the word is solved.

Partial rewards are applied at turn boundreis while the terminal reward is always at the end of the episode.

A system prompt is used to give the model a starting point and encourage it to keep responses within the 1024 token limit.

# Training method

## Rollout generation

The Qwen3 implementation, continoues batching based rollout generator and training algorithem are all implement from scratch with jax and open source here: https://github.com/gabe00122/valm.

Continous batching inference is compiled into a jax jit step so that tokens are generated in a `jax.lax.for_loop` until a entire turn is ready. Despite being relatively simple for short context length batched inference throughput rivals vllm. Slots are kept in vram until the episode is complete which means prompt caching requires no load/offload from vram but the downside of this is that it requires the environments to be fast to not leave idle slots for long. In order to keep the environments as fast as possible they are implemented in rust.

## Loss Function

Standard GAE is applied but with fresh values recalculated in the loss function rather then saved from the rollout generation like typical ppo. Policy loss is masked to tokens generated by the model (so not prompt tokens), while value loss propogates and is bootstrapped through both prompt and policy tokens. When importance sampling correction is used the IS ratio's are claimed to 1.0 for prompt tokens as well.

## Value warmup

The value network is warmup up on frozen policy waits so that online learning begins with value estimates in a reasonable range. The value loss is calculated with a HL gauss value head according to Stop Regressing paper.

After the warmup we can generate value approximations but Qwen3 4b Instruct has poor preformace before training around 0-1%
<EpisodeViewer url="/blog/valm/episode-0.json" metric="value" title="An example episode after value warmup but before online learning" />

In the above episode you can see the model dosn't repect constraints and even guesses 6 letter words at times.

## Online learning

Data from the rollout generation is passed though a circular buffer which yields batches to the update function when it has accumlated a full update batch worth of episodes. The update function uses the same model weights as rollout generation saving vram. In the future this arcetecture could be extended to passing rollout generation over a network transport but for now it's optimized for single device training.

## GRPO Baseline

GRPO is implemented in the same framework as for comparison. Episodes are assigned to groups with each group in the episode getting the same seed and in the case of wordle same hidden word. Because episodes from different groups may finish out of order the group data is interleved/out of order and most be reconstructed with a alternative buffer that uses group id to accumulate groups. When a episode finishes that slot begins on the next available group id.

# Results

## Training metrics

## Preformace

Not only does the solve rate go up, but the model also learns to use the response context as a sort of scratchpad space to search through letter permutations based on the known constraints. This scratchpad strategy evolves over the course of training, and it's not entirely clear whether it is important to the policy or a vestigial artifact. Not only does value approximation lead to learning, but bootstrapped returns are also more effective than Monte Carlo returns.

## Experiements

TODO:
Test MSE over HL gauss

## Episode viewer

<EpisodeViewer url="/blog/valm/episode-378152.json" metric="value" title="A Wordle rollout." />

## Related works
POISE

## Next steps

While this demonstraits value transformers can be used to fine tune a model for wordle, wordle itself might differ significntly from other environments particularly these with long context lengths. Bootstraping needs to be proven on both harder and environments and with longer contexts. 

The value nets for llms could be taken a step further by treating latents as a high dimentional action space and using a SAC style pathwise gradient to train that directly instead of tokens log probs.

# References

- [LST: Ladder Side-Tuning for Parameter and Memory-Efficient Transfer Learning](https://arxiv.org/abs/2206.06522)
- [Does Representation Matter? Exploring Intermediate Layers in Large Language Models](https://arxiv.org/html/2412.09563v1)
- [Stop Regressing: Training Value Functions via Classification for Scalable Deep RL](https://arxiv.org/abs/2403.03950)
- [Your Language Model is Its Own Critic: Reinforcement Learning with Value Estimation from Actor's Internal States](https://arxiv.org/abs/2605.07579)
- [GRPO](https://arxiv.org/pdf/2402.03300)
- [POISE](https://arxiv.org/pdf/2605.07579)
- [DeepSeekMath](https://arxiv.org/pdf/2402.03300)
- [What's Behind PPO's Collapse in Long-CoT?](https://arxiv.org/pdf/2503.01491)
- [Learning Without Critics? Revisiting GRPO in Classical Reinforcement Learning Environments](https://arxiv.org/pdf/2511.03527)
- [REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization](https://arxiv.org/abs/2501.03262)
- [Probing Hidden States for Calibrated, Alignment-Resistant Predictions in LLMs](https://www.medrxiv.org/content/10.1101/2025.09.17.25336018v1)
