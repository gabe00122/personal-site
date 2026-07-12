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

Some of the earliest applications of RL to LLMs with RLHF were based on PPO, with [InstructGPT](https://arxiv.org/abs/2203.02155) being explicitly PPO-based. Since then, [GRPO](https://arxiv.org/abs/2402.03300) has removed the critic and popularized LLM RL reasoning training at scale, becoming the dominant baseline. More recently, the literature has split between a "critic-free is enough" camp ([GRPO](https://arxiv.org/abs/2402.03300), [RLOO](https://arxiv.org/abs/2402.14740), and [GSPO](https://arxiv.org/abs/2507.18071)) and a "bring the critic back" camp ([VC-PPO](https://arxiv.org/abs/2503.01491), [VAPO](https://arxiv.org/abs/2504.05118), [AsyPPO](https://arxiv.org/abs/2510.01656) mini-critics, and [GenAC](https://arxiv.org/abs/2604.10701)).

The thesis of this post is that early LLM PPO implementations were limited by the quality of their value estimates, but value approximation itself remains a promising technique if the architecture can make value prediction cheap and accurate.

GRPO has the advantage of not requiring a critic, but it can only generate a sequence-level baseline, and it does so at the cost of sacrificing prompt diversity. Actor-critics are proven methods in other domains of RL, and GRPO struggles in comparison to PPO on [classic RL environments](https://arxiv.org/abs/2511.03527). LLM reasoning is commonly formalized as a contextual bandit, but increasingly agents are used in complex multi-step environments with uncertainty over environment state. These practical environments could be treated as bandits, but this sacrifices their temporal structure. GRPO structurally assumes the bandit framing with its single scalar advantage per rollout. Value-approximation-based methods can calculate the value for any partial sequence, utilize bootstrapping, and take advantage of the temporal structure of the POMDP.

My experiment fine-tunes Qwen3 on Wordle using a small transformer-based value function that consumes representations from the base model at multiple layers. This architecture provides a critic that adds little overhead compared with the policy update while providing a more granular baseline than sequence-level group normalization. Bootstrapping is shown to outperform Monte Carlo returns with this critic architecture. PPO with this method is able to train Qwen3-4B-Instruct-2507 from a 1% solve rate on Wordle to 99% in 11 hours on an RTX 5090 GPU with no SFT warmup and no KL to reference policy. GRPO, by contrast, fails to push solve rates past 1% without a warm start from a checkpoint with less sparse rewards (similar to the common practice of SFT warmup).

This method is closest to [POISE](https://arxiv.org/abs/2605.07579) - concurrent work I arrived at independently - which predicts a sequence-level baseline from the policy's hidden states; the difference here is a token-level value from a learned ladder side model, trained with bootstrapping rather than Monte Carlo.

## Why critics fell out of favor

Historically, learned value functions for LLM PPO have usually taken one of two forms. The first is to train a value model at roughly LLM scale, which is expensive. The second is to attach a small value head to the final hidden state before the language-model projection. This is much cheaper, but it creates a different problem: the final hidden state is tightly coupled to the next-token distribution. Because almost every direction in the final latent affects the logits, attaching a value loss there risks fighting the policy objective while also not using the most useful representation within the base model for calculating a belief state.

GRPO, on the other hand, has a different failure mode: when a partial reward is dense but a larger reward is sparse, the group normalization gives the dense reward more weight than the larger but sparse reward, creating a bias that can produce strong attractors.

## Architecture

A growing body of evidence suggests that intermediate layers, not the final hidden state, provide the best representations for downstream prediction ([Does Representation Matter](https://arxiv.org/abs/2412.09563); [PING](https://www.medrxiv.org/content/10.1101/2025.09.17.25336018v1)). My approach is to take latent state from multiple layers of the base model and map it to an independent but scaled-down transformer trained on value prediction.

Training an existing policy with a randomly initialized value network could be harmful to the policy ([VC-PPO](https://arxiv.org/abs/2503.01491)), since the value may be very far from the true mean return. One solution is to warm up the value function offline on a frozen policy before online learning; VC-PPO calls this value pretraining. However, with a frozen base model, this gives the value network no way to integrate history and context in a way that differs from the base model's attentional patterns. Being able to learn a function of history is necessary in POMDPs because they require accumulating information about hidden state. In Wordle, for example, the value is an estimate of how constrained the hidden word possibilities are based on all previous guesses and feedback.

This is effectively [ladder side-tuning](https://arxiv.org/abs/2206.06522) with respect to the value function and LoRA training with respect to the policy: like LST, the value stream reads intermediate activations through gated connections and needs no backpropagation through the frozen backbone. Also, similarly to ladder side-tuning, layers from the side network are dropped for efficiency.

<ArchitectureDiagram />

The two streams are:

**Base layer** - a Qwen3 layer with a LoRA adapter. Stacked together, these form the policy, with the residual stream running from the token embedding up to the token prediction.

**Value layer** - a scaled-down version of the Qwen3 architecture with its own attention layers. These form the parallel value network, running from the last reward up to the value prediction.

**Value encode** - every n-th base latent is projected into the value stream through a stop-gradient SwiGLU block, so gradients from the value loss never flow back into the base model. I use a SwiGLU-style MLP block with normalization so the value network can selectively filter latents from the base model into its residual stream. This is to prevent the residuals from the base model from drowning out the signal in the value network's residual stream; the gating in SwiGLU allows existing information to be protected.

### Stop Gradient Risks
The stop gradients allow us to use separate Adam optimizers for value and policy and protect the policy from the value network's gradients, but they create the limitation that the value network cannot request information from the base model; it can only observe what happens to be there. This risk is hopefully mitigated by (1) taking latents from multiple and potentially redundant locations and (2) providing the value network with its own attention layers.

## Environment

Although it could be applied to different tasks, I primarily tested this method on Wordle. Wordle here is defined as a POMDP rather than a contextual bandit. A transition is a single token, not a turn. An episode is multiple turns (up to 6 guesses) capped at 1024 tokens.

### Reward

The reward function is split into two components, both designed so the maximum return is 1.0.

Partial credit, granted once per slot, the first time a slot is revealed:
* +0.025 the first time slot i is yellow-or-green
* +0.025 the first time slot i is green

Each of the 5 slots can contribute at most 0.05, so partial credit tops out at 0.25.

Terminal bonus: +0.75 when the word is solved.

Partial rewards are applied at turn boundaries while the terminal reward is always at the end of the episode.

A system prompt is used to give the model a starting point and encourage it to keep responses within the 1024 token limit.

For GRPO, a full game is considered one rollout, so all the per-turn rewards are rolled into a single scalar reward.

## Rollout generation

The Qwen3 implementation, continuous-batching-based rollout generator, and training algorithm are all implemented from scratch with JAX and open-sourced here: https://github.com/gabe00122/valm.

Continuous batching inference is compiled into a JAX JIT step so that tokens are generated in a `jax.lax.fori_loop` until an entire turn is ready. Despite the implementation being relatively simple, for short context lengths its batched inference throughput rivals vLLM (~8000 tokens/s). Slots are kept in VRAM until the episode is complete, which means prompt caching requires no load/offload from VRAM, but the downside of this is that it requires the environments to be fast so they do not leave idle slots for long. In order to keep the environments as fast as possible (and because it's fun!), they are implemented in Rust.

## Loss Function

Standard GAE is applied, but with fresh values recalculated in the loss function rather than saved during rollout generation, as is typical in PPO. I only ever use one epoch, but the values might still be stale because of continuous batching. This, however, means that the value network is only used in the loss function and does not impact rollout generation. The policy loss is masked to tokens generated by the model (so not prompt tokens), while the value loss propagates and is bootstrapped through both prompt and policy tokens. A discount of 1.0 is applied to most tokens, but at the turn boundary, the discount is set to 0.97 for one token to encourage finishing the game in fewer turns.

## Value warmup

The value network is warmed up on frozen policy weights so that online learning begins with value estimates in a reasonable range. Because the values are calculated fresh within the loss function on every update the same loss function can be used for both value warmup and online learning. The value loss is calculated with an HL-Gauss value head, following [Stop Regressing](https://arxiv.org/abs/2403.03950), which itself demonstrated transformer value functions on Wordle, a precedent for both choices here. It's notable that the Stop Regressing paper trained a transformer-based model on Wordle specifically and saw a 43% gain over MSE.

After the warmup we can generate value approximations, but Qwen3 4B Instruct has poor performance before training, around a 0-1% solve rate.
<EpisodeViewer
    metric="value"
    episodes={[
  		{ url: "/blog/valm/vt-0.json", label: "One" },
  		{ url: "/blog/valm/vt-1.json", label: "Two" },
      { url: "/blog/valm/vt-2.json", label: "Three" },
      { url: "/blog/valm/vt-3.json", label: "Four" }
    ]}
  />

In the episodes above, you can see that the model doesn't respect constraints and even guesses 6-letter words at times.

## Online learning

Data from the rollout generation is passed through a circular buffer that yields batches to the update function when it has accumulated a full update batch worth of episodes. The update function uses the same model weights as rollout generation, saving VRAM. In the future, this architecture could be extended to passing rollout generation over a network transport, but for now it's optimized for single-device training.

## GRPO Baseline

GRPO is implemented in the same framework for comparison. Episodes are assigned to groups, with each episode in the group receiving the same seed and, in the case of Wordle, the same hidden word. Because episodes from different groups may finish out of order, the group data as it hits the data pipe is interleaved and must be reconstructed with an alternative buffer that uses group ID to accumulate groups. When an episode finishes, that env/slot begins on the next available group ID.
Because GRPO uses undiscounted returns, its objective is different from the turn-discounted objective PPO uses. This means PPO is incentivized to complete the problem with as few turns as possible, while GRPO is incentivized to complete the problem in any number of turns within the six-turn limit.

## Experiment

With three seeds per training config, I tested PPO with the value transformer and HL-Gauss, PPO without the value transformer and with a last-latent probe only, MSE instead of the HL-Gauss head, and Monte Carlo returns with lambda 1.0 versus 0.95 in the other experiments.
I also tested GRPO starting from the base model (GRPO-cold) and GRPO starting from a value transformer checkpoint at a 10% solve rate (GRPO-warm).

The same policy (and value, where applicable) learning rates were used for all of these models. It is possible that some of these runs would have worked better with more tuning. In particular, I suspect Monte Carlo learning might just need the learning rate reduced to better adapt to the higher-variance advantages.

**Shared settings** (all runs unless noted below):

| | Setting | Value |
|---|---|---|
| **Model** | Base model | Qwen3-4B-Instruct-2507 |
| | Max sequence length | 1024 |
| **Policy (LoRA)** | Rank | 64 |
| | Applied to | Attention + MLP |
| | Optimizer | AdamW, lr 4e-5, Î²â‚ 0.9, Î²â‚‚ 0.98, wd 0.01 |
| | Schedule | Cosine, 10% warmup |
| | Grad norm clip | 1.0 |
| **Value transformer** | Layers | 12 |
| | Embed dim | 256 |
| | Attention | 8 heads (8 KV), head dim 32 |
| | MLP width | 512 |
| | Latent encoder rank | 256 |
| | Optimizer | AdamW, lr 1e-4, β₁ 0.9, β₂ 0.98, wd 0.01 |
| **PPO loss** | Clip range | 0.2 low / 0.28 high |
| | GAE λ | 0.95 |
| | Discount | 1.0 (0.97 at turn boundaries) |


**Per-method differences:**

| Run | Differs from shared config |
|---|---|
| VT + HL-Gauss | — (reference config). HL-Gauss head: 51 bins, σ 0.02, support [−0.1, 1.1] |
| VT + MSE | MSE value head |
| Last-latent only | Value net reads only the final hidden state; 0 value transformer layers |
| Monte Carlo | GAE λ = 1.0 (turn λ also 1.0) |
| GRPO | No critic; group size 8; group-normalized sequence-level advantage |

### Speed
When benchmarking the update function (the only place GRPO and PPO differ in this implementation), GRPO is only 6% faster than PPO with the ~25.8M parameter value transformer. This is because the latents used to calculate the policy loss can be reused to calculate the value loss, and the value transformer is tiny compared to the base model, being only 0.6% of total parameters.

<Image url="/blog/valm/vs_vt.webp" description="" alt="" />

Looking at solve rate, HL-Gauss versus MSE was a null result. I was not able to reproduce the result from the Stop Regressing paper (on the same environment!).


Every model trained with the value transformer was ahead of every model trained with the last latent only (although there was one seed that came very close). The value transformer learned in a consistent band, while last-latent-only had significantly more per-seed variance. Monte Carlo learning with lambda 1.0 diverged completely in 50% of runs; here I have just plotted the survivors (3 out of 6). GRPO from a cold start gets stuck maximizing partial credit rewards and never learns to solve the game consistently. From a warm start, GRPO consistently learns but underperforms the value transformer.

<Image url="/blog/valm/vs_vt_reward.webp" description="" alt="" />
Total reward tells a similar story, although you can see that cold-start GRPO improves on partial credit but not on solve rate.


<Image url="/blog/valm/vs_vt_turns.webp" description="Turns taken per game, failures are counted as six" alt="" />

Here, when plotting turns taken per game, HL-Gauss separates from MSE somewhat, although I'm not sure if it's significant. Unlike PPO, where turns taken per game decreases as solve rate increases, GRPO stays relatively steady. This reflects the fact that GRPO didn't use discounting at turn boundaries, so it was actually incentivized to maximize information before solving the puzzle because this was the safest strategy.

<Image url="/blog/valm/ev_mean_bands.webp" description="" alt="" />

Comparing the value transformer and the last-latent-only model on explained variance, particularly at the end of training, the value transformer's critic had much higher explained variance.

### Episode patterns

Here are a few episodes from the end of value-transformer training.

<EpisodeViewer
  metric="value"
  episodes={[
		{ url: "/blog/valm/vt-399996.json", label: "One" },
		{ url: "/blog/valm/vt-399997.json", label: "Two" },
    { url: "/blog/valm/vt-399998.json", label: "Three" },
    { url: "/blog/valm/vt-399999.json", label: "Four" }
  ]}
/>

You can see the value steps up on the turn-level discount and steps down when the partial reward is banked. The model will sometimes output none answer tokens in context before committing to an option. Notably, this context strategy resembles enumerating options under the current constraints, but they are often not real words until the model commits to a guess.


## Next steps

While this demonstrates that value transformers can be used to fine-tune a model for Wordle, Wordle itself might differ significantly from other environments, particularly those with long context lengths. Bootstrapping needs to be demonstrated on both harder environments and with longer contexts.

The value nets for LLMs could be taken a step further by treating latents as a high-dimensional action space and using a SAC-style pathwise gradient to train some of the latents used by the value network as the continuous actions.

# References

- [LST: Ladder Side-Tuning for Parameter and Memory-Efficient Transfer Learning](https://arxiv.org/abs/2206.06522)
- [Does Representation Matter? Exploring Intermediate Layers in Large Language Models](https://arxiv.org/html/2412.09563v1)
- [Stop Regressing: Training Value Functions via Classification for Scalable Deep RL](https://arxiv.org/abs/2403.03950)
- [Your Language Model is Its Own Critic: Reinforcement Learning with Value Estimation from Actor's Internal States — POISE](https://arxiv.org/abs/2605.07579)
- [DeepSeekMath (introduces GRPO)](https://arxiv.org/abs/2402.03300)
- [What's Behind PPO's Collapse in Long-CoT? (VC-PPO)](https://arxiv.org/pdf/2503.01491)
- [Learning Without Critics? Revisiting GRPO in Classical Reinforcement Learning Environments](https://arxiv.org/pdf/2511.03527)
- [InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [RLOO: Back to Basics — Revisiting REINFORCE-style Optimization for RLHF](https://arxiv.org/abs/2402.14740)
- [GSPO: Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
- [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)
- [AsyPPO: Asymmetric Proximal Policy Optimization — mini-critics boost LLM reasoning](https://arxiv.org/abs/2510.01656)
- [GenAC: Bringing Value Models Back — Generative Critics for Value Modeling in LLM RL](https://arxiv.org/abs/2604.10701)
