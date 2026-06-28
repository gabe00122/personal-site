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

Some of the earliest applications of RL to LLMs with RLHF were based on PPO, with [InstructGPT](https://arxiv.org/abs/2203.02155) being explicitly PPO-based. Since then, [GRPO](https://arxiv.org/abs/2402.03300) has removed the critic and popularized LLM RL reasoning training at scale, becoming the dominant baseline. More recently, the literature has split between a "critic-free is enough" camp ([GRPO](https://arxiv.org/abs/2402.03300), [RLOO](https://arxiv.org/abs/2402.14740), [GSPO](https://arxiv.org/abs/2507.18071), and [REINFORCE++](https://arxiv.org/abs/2501.03262)) and a "bring the critic back" camp ([VC-PPO](https://arxiv.org/abs/2503.01491), [VAPO](https://arxiv.org/abs/2504.05118), [AsyPPO](https://arxiv.org/abs/2510.01656) mini-critics, and [GenAC](https://arxiv.org/abs/2604.10701)).

The thesis of this post is that early LLM PPO implementations were limited by the quality of their value estimates, but value approximation itself remains a promising technique if the architecture can make value prediction cheap and accurate.

GRPO has the advantage of not requiring a critic, but it can only generate a sequence-level baseline, and it does so at the cost of sacrificing prompt diversity; the trade-off [REINFORCE++](https://arxiv.org/abs/2501.03262) makes precise with its single-rollout, globally-normalized advantage. Actor-critics are proven methods in other domains of RL and GRPO struggles in comparison to PPO on [classic RL environments](https://arxiv.org/abs/2511.03527). Even in an LLM context, especially when concerning POMDPs rather than contextual bandits, in principle, value approximation methods can leverage a model's own representation to form an accurate baseline for every partial sequence and not just a scalar for the entire turn.  

My experiment fine-tunes Qwen3 on Wordle using a small transformer-based value function that consumes representations from the base model at multiple layers. This architecture provides a critic that adds little overhead compared with the policy update while providing a more granular baseline than sequence-level group normalization. Contrary to other work, bootstrapping is shown to outperform Monte Carlo returns with this critic architecture. I'm able to train Qwen3-4B-Instruct-2507 from a 1% solve rate on Wordle to 99% in 11 hours on an RTX 5090 GPU.

This method is closest to [POISE](https://arxiv.org/abs/2605.07579) - concurrent work I arrived at independently - which predicts a sequence-level baseline from the policy's hidden states; the difference here is a token-level value from a learned ladder side model, trained with bootstrapping rather than Monte-Carlo.

# Why critics fell out of favor

Historically, learned value functions for LLM PPO have usually taken one of two forms. The first is to train a value model at roughly LLM scale, which is expensive. The second is to attach a small value head to the final hidden state before the language-model projection. This is much cheaper, but it creates a different problem: the final hidden state is tightly coupled to the next-token distribution. Because almost every direction in the final latent affects the logits, attaching a value loss there risks fighting the policy objective while also not using the most useful representation within the base model for long-horizon prediction.

# Architecture

A growing body of evidence suggests that intermediate layers, not the final hidden state, provide the best representations for downstream prediction ([Does Representation Matter](https://arxiv.org/abs/2412.09563); [PING](https://www.medrxiv.org/content/10.1101/2025.09.17.25336018v1)). My approach is to take latent state from multiple layers of the base model and map it to an independent but scaled-down transformer trained on value prediction.

Training an existing policy with a randomly initialized value network could be harmful to the policy ([VC-PPO](https://arxiv.org/abs/2503.01491)), since the value may be very far from the true mean return. One solution is to warm up the value function offline on a frozen policy before online learning; VC-PPO calls this value pretraining. However, with a frozen base model, this gives the value network no way to integrate history and context in a way that differs from the base model's attentional patterns. Being able to learn a function of history is particularly important for value prediction because it is a long-term forecast of the future and is highly dependent on hidden environment state. This is why it is useful for the value network to have its own attention layers.

This is effectively [ladder side-tuning](https://arxiv.org/abs/2206.06522) with respect to the value function and LoRA training with respect to the policy: like LST, the value stream reads intermediate activations through gated connections and needs no backpropagation through the frozen backbone. Also, similarly to ladder side-tuning, layers from the side network are dropped for efficiency.

<ArchitectureDiagram />

The two streams are:

**Base layer** - a Qwen3 layer with a LoRA adapter. Stacked together, these form the policy, with the residual stream running from the token embedding up to the token prediction.

**Value layer** - a scaled-down version of the Qwen3 architecture with its own attention layers. These form the parallel value network, running from the last reward up to the value prediction.

**Value encode** - every n-th base latent is projected into the value stream through a stop-gradient SwiGLU block, so gradients from the value loss never flow back into the base model. I use a SwiGLU-style MLP block with normalization so the value network can selectively filter latents from the base model into its residual stream.

# Environment

Although it could be applied to different tasks, I primarily tested this method on Wordle. Wordle here is defined as a POMDP rather than a contextual bandit. A transition is a single token, not a turn. An episode is multiple turns (up to 6 guesses) capped at 1024 tokens.

## Reward

The reward function is split into two components, both designed so the maximum return is 1.0.

Partial credit, granted once per slot, the first time a slot is revealed:
* +0.025 the first time slot i is yellow-or-green
* +0.025 the first time slot i is green

Each of the 5 slots can contribute at most 0.05, so partial credit tops out at 0.25.

Terminal bonus: +0.75 when the word is solved.

Partial rewards are applied at turn boundaries while the terminal reward is always at the end of the episode.

A system prompt is used to give the model a starting point and encourage it to keep responses within the 1024 token limit.

For GRPO, a full game is considered one rollout and so all the per-turn rewards are rolled into a single scalar reward. 

## Rollout generation

The Qwen3 implementation, continuous-batching-based rollout generator and training algorithm are all implemented from scratch with JAX and open-sourced here: https://github.com/gabe00122/valm.

Continuous batching inference is compiled into a JAX JIT step so that tokens are generated in a `jax.lax.for_loop` until an entire turn is ready. Despite being relatively simple for short context length, batched inference throughput rivals vLLM (~8000 tokens/s). Slots are kept in VRAM until the episode is complete, which means prompt caching requires no load/offload from VRAM but the downside of this is that it requires the environments to be fast so they do not leave idle slots for long. In order to keep the environments as fast as possible they are implemented in Rust.

## Loss Function

Standard GAE is applied, but with fresh values recalculated in the loss function rather than saved from the rollout generation like typical PPO. This, however, means that the value network is only used in the loss function and does not impact rollout generation. Policy loss is masked to tokens generated by the model (so not prompt tokens), while value loss propagates and is bootstrapped through both prompt and policy tokens. When importance sampling correction is used the IS ratios are clamped to 1.0 for prompt tokens as well. A discount of 1.0 is applied to most tokens but at the turn boundary discount is set to 0.97 for one token to encourage finishing the game in future turns. 

## Value warmup

The value network is warmed up on frozen policy weights so that online learning begins with value estimates in a reasonable range. The value loss is calculated with an HL-Gauss value head, following [Stop Regressing](https://arxiv.org/abs/2403.03950), which itself demonstrated transformer value functions on Wordle, a precedent for both choices here. It's notable that the Stop Regressing paper trained a transformer-based model on Wordle specifically and saw a 43% gain over MSE.

After the warmup we can generate value approximations but Qwen3 4B Instruct has poor performance before training, around 0-1%
<EpisodeViewer
    metric="value"
    episodes={[
  		{ url: "/blog/valm/episode-1.json", label: "One" },
  		{ url: "/blog/valm/episode-2.json", label: "Two" },
      { url: "/blog/valm/episode-3.json", label: "Three" },
      { url: "/blog/valm/episode-4.json", label: "Four" }
    ]}
  />

In the above episode you can see the model doesn't respect constraints and even guesses 6 letter words at times.

## Online learning

Data from the rollout generation is passed through a circular buffer that yields batches to the update function when it has accumulated a full update batch worth of episodes. The update function uses the same model weights as rollout generation, saving VRAM. In the future, this architecture could be extended to passing rollout generation over a network transport but for now it's optimized for single device training.

## GRPO Baseline

GRPO is implemented in the same framework for comparison. Episodes are assigned to groups with each episode in the group receiving the same seed and, in the case of Wordle, the same hidden word. Because episodes from different groups may finish out of order, the group data as it hits the data pipe is interleaved and must be reconstructed with an alternative buffer that uses group ID to accumulate groups. When an episode finishes, that env/slot begins on the next available group ID.

# Results

When benchmarking the update function (the only place GRPO and PPO differ in this implementation) GRPO is 6% faster than PPO with the ~25.8M parameter value transformer. This is because the latents used to calculate the policy loss can be reused to calculate the value loss, and the value transformer is tiny compared to the base model, being only 0.6% of total parameters.

## Training metrics

## Performance

Not only does the solve rate go up, but the model also learns to use the response context as a sort of scratchpad space to search through letter permutations based on the known constraints. This scratchpad strategy evolves over the course of training, and it's not entirely clear whether it is important to the policy or a vestigial artifact. Not only does value approximation lead to learning, but bootstrapped returns are also more effective than Monte Carlo returns. This runs counter to [VC-PPO](https://arxiv.org/abs/2503.01491), which found Monte-Carlo (λ=1) better for the value in long chains of thought, where a single terminal reward decays before it reaches early tokens. Wordle is short and densely rewarded at every turn boundary, so that decay never bites and bootstrapping's variance reduction wins, which makes long-context environments the place this could flip.

## Experiments

TODO:
Test MSE over HL-Gauss

## Episode viewer

<EpisodeViewer
  metric="value"
  episodes={[
		{ url: "/blog/valm/episode-380001.json", label: "One" },
		{ url: "/blog/valm/episode-380002.json", label: "Two" },
    { url: "/blog/valm/episode-380003.json", label: "Three" },
    { url: "/blog/valm/episode-380004.json", label: "Four" }
  ]}
/>

## Next steps

While this demonstrates value transformers can be used to fine-tune a model for Wordle, Wordle itself might differ significantly from other environments, particularly those with long context lengths. Bootstrapping needs to be proven on both harder environments and with longer contexts. 

The value nets for LLMs could be taken a step further by treating latents as a high-dimensional action space and using a SAC-style pathwise gradient to train that directly instead of token log-probs.

# References

- [LST: Ladder Side-Tuning for Parameter and Memory-Efficient Transfer Learning](https://arxiv.org/abs/2206.06522)
- [Does Representation Matter? Exploring Intermediate Layers in Large Language Models](https://arxiv.org/html/2412.09563v1)
- [Stop Regressing: Training Value Functions via Classification for Scalable Deep RL](https://arxiv.org/abs/2403.03950)
- [Your Language Model is Its Own Critic: Reinforcement Learning with Value Estimation from Actor's Internal States — POISE](https://arxiv.org/abs/2605.07579)
- [DeepSeekMath (introduces GRPO)](https://arxiv.org/abs/2402.03300)
- [What's Behind PPO's Collapse in Long-CoT? (VC-PPO)](https://arxiv.org/pdf/2503.01491)
- [Learning Without Critics? Revisiting GRPO in Classical Reinforcement Learning Environments](https://arxiv.org/pdf/2511.03527)
- [REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization](https://arxiv.org/abs/2501.03262)
- [InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [RLOO: Back to Basics — Revisiting REINFORCE-style Optimization for RLHF](https://arxiv.org/abs/2402.14740)
- [GSPO: Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
- [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)
- [AsyPPO: Asymmetric Proximal Policy Optimization — mini-critics boost LLM reasoning](https://arxiv.org/abs/2510.01656)
- [GenAC: Bringing Value Models Back — Generative Critics for Value Modeling in LLM RL](https://arxiv.org/abs/2604.10701)
- [Probing Hidden States for Calibrated, Alignment-Resistant Predictions in LLMs](https://www.medrxiv.org/content/10.1101/2025.09.17.25336018v1)
