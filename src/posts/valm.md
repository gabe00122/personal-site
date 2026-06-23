---

title: Value functions for LLMs
description: Value functions for LLM fine-tuning
date: '2026-06-12'
published: true
---------------

<script>
    import VideoPlayer from "$lib/components/video.svelte";
    import Image from "$lib/components/image.svelte";
    import EpisodeViewer from "$lib/components/episode/episodeViewer.svelte";
</script>

# Introduction

Some of the earliest applications of RL to LLMs with RLHF were based on PPO, with InstructGPT being explicitly PPO-based. Since then, GRPO has removed the critic and popularized pure RL reasoning training at scale, becoming the dominant baseline. More recently, the literature has split between a "critic-free is enough" camp (GRPO, RLOO, GSPO, and REINFORCE++) and a "bring the critic back" camp (VC-PPO, VAPO, AsyPPO mini-critics, and GenAC).

The core thesis of this post is that early LLM PPO implementations were limited by the quality of their value estimates, but value approximation itself remains a promising technique if the architecture can make value prediction cheap and accurate.

**Point out GRPO's struggles in classic RL environments (https://arxiv.org/pdf/2511.03527).**

GRPO has the advantage of not requiring a critic, but it can only generate a sequence-level baseline, and it does so at the cost of sacrificing prompt diversity. By contrast, a good value function doesn't just learn the expected return for an entire sequence; it is a learned heuristic for every partial sequence. In principle, that heuristic can use information from the model’s own representation of the state, learn from a single rollout without a group, and utilize information from state transitions through bootstrapping.

My experiment fine-tunes Qwen3 on Wordle using a small value model that consumes representations from the base model. This architecture provides a critic that adds little overhead compared with the policy update while providing a more granular baseline than sequence-level group normalization.

# Approach

Historically, learned value functions for LLM PPO have usually taken one of two forms. The first is to train a value model at roughly LLM scale, which is expensive. The second is to attach a small value head to the final hidden state before the language-model projection. This is much cheaper, but it creates a different problem: the final hidden state is tightly coupled to the next-token distribution. Because almost every direction in the final latent affects the logits, attaching a value loss there risks fighting the policy objective.

A growing body of evidence suggests that latents from deeper layers of the base model provide better representations for downstream predictions. PING (https://www.medrxiv.org/content/10.1101/2025.09.17.25336018v1) I don't know which layers are most useful, so my approach was to take every n-th latent and map it to a smaller value prediction model.

Training an existing policy with a randomly initialized value network could be harmful to the policy (VC-PPO), since the value may be very far from the true mean return. One solution is to warm up the value function offline on a frozen policy before online learning. However, with a frozen base model, this gives the value network no way to integrate history and context in its own learned way. Being able to learn a function of history is particularly important for value prediction because it is a long-term forecast of the future and is highly dependent on the environment state.

This is why I designed the value network as a separate, parallel, ladder-style transformer that consumes latents. The attention patterns for the value network can be learned during the offline critic warmup without affecting the base model's attention.

```
 Last Token Embedding                     Last Reward
     |                                        |
     |----------------------|                 |
     |                 Value Encode           |
     v                      |                 |
  Base Layer 1              +<--Reward Encode--
     |                      |
     |                      v
     |--Value Encode-->Value Layer 1
     v                      |
  Base Layer 2              |
     |                      |
     |                      |
     |                      |
     v                      |
  Base Layer 3              |
     |                      |
     |   (taken every n)    v
     |--Value Encode-->Value Layer 2
     v                      |
  Base Layer 4              |
     |                      |
  Token Decode         Value Prediction


Base Layer:
Qwen3 layer with a LoRA adapter

Value Layer:
Scaled-down version of the Qwen3 architecture with its own attention layers

Value Encode:
x = jax.lax.stop_gradient(x)
x = self._normalize(x)
gate = self._up_gate(x)
x = self._encode_up(x)
x = jax.nn.silu(x) * gate
x = self._encode_down(x)

```

I use a swiglu style mlp block with normalization so the value network can selectively filter latents from the base model into it's residual stream.

...

## Implementation

Wordle was used as a test environment to explore this method. It is a simple text-based game involving uncertainty and constraint resolution. While more work needs to be done to compare this method with other approaches, it seems to be promising and efficient. I'm able to train Qwen3-4B-Instruct-2507 from a 1% solve rate on Wordle to 99% in 11 hours on a single consumer GPU. Not only does the solve rate go up, but the model also learns to use the response context as a sort of scratchpad space to search through letter permutations based on the known constraints. This scratchpad strategy evolves over the course of training, and it's not entirely clear whether it is important to the policy or a vestigial artifact. Not only does value approximation lead to learning, but bootstrapped returns are also more effective than Monte Carlo returns.

**Note wordle is a POMDP not a contextual bandit**

Things to prove:

* Beats using only the last-layer latent

* Beats REINFORCE with a baseline

* Is computationally cheap—advantages are calculated similarly to IMPALA, with IS ratios rather than PPO, so values are calculated only in the update function, not during inference

* LoRA performance gotcha

* Custom inference

* Continuous batching

* Importance sampling / mask prompts

* Policy loss over policy, value loss through prompts

* HL-Gauss

* Reward encoding

# Results

## Training metrics

## Episode viewer

<EpisodeViewer url="/blog/valm/episode-378152.json" metric="log_probs" title="A Wordle rollout." />

# Technical tricks

## `lax.scatter` for KV-cache updates

## Continuous batching in a compiled JAX function

## Circular data buffer

## Papers

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
