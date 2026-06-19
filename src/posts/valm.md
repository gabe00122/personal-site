---
title: Value functions for LLMs
description: Value functions for LLM fine tuning
date: '2026-06-12'
published: true
---

<script>
    import VideoPlayer from "$lib/components/video.svelte";
    import Image from "$lib/components/image.svelte";
</script>

# Introduction

Some of the earliest applications of RL to LLMs with rlhf were based on ppo, with instructgpt being explicitly ppo based.
Since then GRPO removed the critic and showed and popularized pure RL reasoning training at scale becoming the dominant baseline. More recently the literature has split between a "critic-free is enough" (GRPO, RLOO, GSPO, and REINFORCE++) and a "bring the critic back" (VC-PPO, VAPO, AsyPPO mini-critics and GenAC) camp.

The core thesis of this post is that early LLM PPO implementations were limited by the quality of the value estimation but that value approximation itself remains a promising technique if the arcetecture were to make value prediction cheap and accurate.

**Point out GRPO stuggle in classic RL environments (https://arxiv.org/pdf/2511.03527).**

GRPO has the advantage of not requireming a critic but can only generate a sequence level baseline and it does so at the cost of sacrificing prompt diversity. By contrast a good value function dosn't just learn the expected return for a entire sequence it is a learned heuristic for every partial sequence. In principle, that heuristic can use information from the model’s own representation of the state, not just the final reward of a sampled group. **I need to work in value approx can learn from a single rollout**

My experiment fine-tunes Qwen3 on Wordle using a small value model that consumes representations from the base model. The result is a critic that adds little overhead compared with the policy update, while providing a more granular baseline than sequence-level group normalization.

GRPO gives each completion a baseline relative to other completions from the same prompt. That is powerful, but it pays for variance reduction by sacrificing prompt diversity to create groups. A learned value function can amortize information across episodes, assign different baselines to different token positions, and learn from states transition through bootstrapping.

# Approach
Historically, learned value functions for LLM PPO have usually taken one of two forms. The first is to train a value model at roughly LLM scale, which is expensive. The second is to attach a small value head to the final hidden state before the language-model projection. This is much cheaper, but it creates a different problem: The final hidden state is tightly coupled to the next-token distribution. Because almost every direction in the final latent affects the logits, attaching a value loss there risks fighting the policy objective.

A growing body of evidence suggests that latents deeper in the base model provide better representations for downstream predictions, I don't know which layers are most useful so my approach was take take every n-latents and map them to a smaller value prediction model.

Training a existing policy from a randomly initalized value network could be harmful to the policy (VC-PPO) since the value may be very far from the true mean return. A solution is to offline warmup the value function on a frozen policy before online learning. But with a frozen base model this gives the value network no way to integrate history/context in it's own learned way, and being able to learn a function of history is preticular imprortant for value prediction because it's a long term forcast for the future and highly dependent on environment state.

This is why I designed the value network as a seperate parrelel ladder style transformer that consumers latents. The attention patterns for the value network can be learned during the offline critic warmup without effecting the base policy.

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
     |
     |
     |
     v
  Base Layer 3
     |
     |
     |--Value Encode-->Value Layer 2
     v
  Base Layer 4
```

...

## Implimentation
Wordle was used as a test environment to explore this method, it's a simple text based game with uncertently and constraint resolution. While more work needs to be done to compare this to other approaches it seems to be promising and efficent. I'm able to train Qwen3-4B-Instruct-2507 from 1% solve rate on wordle to 97% in 8 hours on a single consumer gpu. Not only does the solve rate go up but the model learns to use the response context as a sort of scratch pad space to search letter permutations based on the known constraints. Not only does value approximation lead to learning but bootstrapped returns are more effective then monte carlo returns.


Things to prove:
* Beats last layer only latent
* Beats reinforce with baseline
* Is computationaly cheap - advantages are calculated like impala with IS ratios and not PPO so values are only calculated in the update function not inference

* Lora preformace gotcha
* custom inference
* contious batching
* Importance sampling / mask prompts
* policy loss over policy, value loss through prompts
* HL Gauss
* reward encoding

# Results

## Training metrics

## Episode viewer


# Techniqual tricks
## lax scatter for kv cache updates
## Contiuous batching in a compiled jax function
## Circluar Data buffer

## Papers
LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning
https://arxiv.org/abs/2206.06522

Does Representation Matter? Exploring Intermediate Layers in Large Language Models
https://arxiv.org/html/2412.09563v1

Stop Regressing: Training Value Functions via Classification for Scalable Deep RL
https://arxiv.org/abs/2403.03950

Your Language Model is Its Own Critic: Reinforcement Learning with Value Estimation from Actor's Internal States
https://arxiv.org/abs/2605.07579

GPRO
https://arxiv.org/pdf/2402.03300
