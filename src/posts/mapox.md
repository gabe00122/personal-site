---
title: Mapox
description: Mapox
date: '2026-03-24'
categories:
  - reinforcement learning
  - jax
  - flax
  - transformer
  - deep learning
published: false
---

# Memory in Multi-Agent Reinforcement Learning
*A blog post based on a presentation by Gabriel Keith*

---

## Introduction: Why Memory Changes Everything

Memory isn't just a performance trick — it's a prerequisite for meaningful communication.

An agent can only *use* what was communicated to it if it can *remember* it. To learn strategies like deception or cooperation, agents need to reason not just over the last observation but over the full history of what they've seen.

**Running example:** An agent scouts unfamiliar territory and guides a partner back to a useful resource. This simple scenario turns out to require surprising machinery to pull off.

---

## The Surprising Result: Emergent Communication Without a Communication Channel

Before diving into the architecture, here's the punchline — because it's the thing worth building toward.

In a two-agent cooperative environment (a rabbit and a turtle, both rewarded when the turtle finds a flag):

- No communication mechanism was designed
- The rabbit learned to **use its own position** within the turtle's field of view as a signal — pointing with its body
- This required the rabbit to remember where both the reward *and* the turtle were
- Because the turtle moves while the rabbit is exploring, the rabbit also had to **model the other agent's behavior** to predict its location on return

This is emergent signaling from memory pressure alone. The rest of this post is about what made it possible.

---

## The Framework: JAX-Based RL at Scale

A custom PPO trainer and environment suite, written in JAX for high throughput:

- **~3.8M steps/second** with a 2-layer network
- **~1.36M steps/second** with an 8-layer network
- Supports both **Transformer** and **GRU**-based memory
- Agents share weights but maintain **separate memory states**

### The LLM Connection

The architecture is closer to a language model than it might seem:

| Similarity | Difference |
|---|---|
| Discrete actions ↔ tokens | Value prediction is required |
| Positional encodings work the same way | Requires encoding last observation + last reward |

This framing turns out to be more than an analogy — see the final section.

---

## Environments

All environments share a common interface (egocentric observation, discrete actions) to support **multi-task learning**. This matters for training stability.

### Find & Return
Search for goal flags in a procedurally-generated maze with destructible walls. Agents are teleported to a new random position after finding a flag — testing persistent spatial memory.

### Scouts
An **asymmetric cooperative** environment with two roles. Multi-task learning was required for stable training — single-task training collapsed.

### Predator Prey
- Prey eats random plants
- Predators eat prey
- Tall grass conceals agents

Food caching mechanics emerge from the transformer's action history serving as implicit location memory.

### King of the Hill
Two-team competitive environment: knights and archers, destructible walls, control point capture, team-shared rewards.

---

## Results

### Scaling Behavior
[Depth and width scaling charts from the presentation — transformer memory benefits more from depth than width in these environments]

### Craftax Benchmark
- **20.8% reward after 1B steps**
- Second highest known score: 18.3%

### Communication Scales with Agent Count
Increasing agent count improved *individual* reward — a sign of genuine emergent coordination rather than just parallelism.

### Trueskill Zero-Sum Comparison
[Competitive environment results — league-based self-play enabled meaningful competitive training]

---

## Other Implementation Details Worth Mentioning

- **Hyperparameter optimization** with Optuna
- **Categorical value head** (rather than scalar regression)
- **Muon optimizer**
- **League-based self-play** for competitive environments

---

## Open Source

- Trainer: [github.com/gabe00122/jaxrl](https://github.com/gabe00122/jaxrl)
- Standalone grid environments: [github.com/gabe00122/mapox](https://github.com/gabe00122/mapox)

---

## What's Next: Applying This to LLMs

The same episodic memory approach is now being applied to fine-tune **Qwen3** with PPO.

Challenges specific to LLMs:
- Action and episode sizes are **jagged** (variable-length), which complicates batching
- Choice of granularity: treat each **token** as an action, or each **conversation turn**?

Early results:
- Wordle success rate: **5% → 55%**

The underlying bet: if transformer memory enables emergent behavior in gridworlds, the same mechanism might enable qualitatively different behavior in language models — where the "environment" is a conversation and communication is the task itself.

---

## Closing Thought

The scout/turtle result is a small thing that points at something large. We didn't design a signaling protocol. We gave an agent enough memory to need one, and it invented one. What else emerges when you stop compressing?
