---
title: Mapox
description: Memory in Multi-Agent Reinforcement Learning
date: '2026-03-24'
categories:
  - reinforcement learning
  - jax
  - flax
  - transformer
  - deep learning
published: false
---

<script>
    import VideoPlayer from "../routes/components/video.svelte";
    import Image from "../routes/components/image.svelte";
    import Collapsible from "../routes/components/collapsible.svelte"
</script>


---

## Memory and Communication

The primary motivation for my projects JaxRL and Mapox was to study the effects of memory arceteectures on emergent communication.
While what I wanted to study was emergent communication I ended up spending most of my time working on memory.

Memory is a strong prerequeists to intresting communicaiton, and I think there's likely a deep connection between the charecteristics of the memory and the charecteristics of the signals between agents.

Memory both lets agents *gather* something that will be useful to communicate and lets agents make *use* of communication later.

Another justification for focusing on memory is that social environments are always pomdps. 
If you are in a enviorment with unknown agents then those agents policies are partilly observable state, if those agents also have memory than that's another axis of partially observable state. So adding memory to a multi agent envirnoment might mean that my memory needs to be used to reason about your memory and inclinations.

I think there's also a connection between communication and exploration. Exploration is about gathering information that will let you make better decisions later, communication can also be about gathering information that will let you make better decisions later. Unlike exploration communication isn't always motovated by improving your own policy (although it certainly can be), you might have a incentive to improve the policy of a agent your cooperating with or in a aversarial setting to degrade the policy of another agent (deception).

It's easy to imagine environments where agents have incentives to model eachothers memory inside their own memory and to act on the memory of another.

All of this is to say I embarct to study communication but I ended up spending a disproportinete amount of my time thinking about memory.

# note

When you add memory to one agent that becomes more partially observable state to the other agents
so adding memory to multiagent RL both gives them a tool to deal with partial observability but also makes it even more partially observable.
I do think a environment where agents had hidden objectives would be both tractable and interesting
they would need to learn if others are seeking to cooperate or not

# note

---

## Cooperative Exploration

To test if agents could share information from their memory I designed a simple asymetric exploration game in my jax based environment collection [mapox](https://github.com/gabe00122/mapox)

The following model used my [jaxrl](https://github.com/gabe00122/jaxrl) framework for training and is saved to hugging face [here](https://huggingface.co/gabe00122/mapox-checkpoints)

There are two agent types, a **fast rabbit** and a **slow turtle** agent. Both agents are rewarded when the **turtle reaches the flag** location.

- Both agents have seperate memory
- Agents have a and egocentric field of view highlighted by the shaded square
- Each episode uses a new map from a map generator
- The rabits only effect on the environment is to observe and to be observed by the turtle

Here's an example episode
<VideoPlayer
    url="/blog/memorycomm/scouts.mp4"
    description=""
    alt="A video of a cart-pole policy after 800,000 steps training"
/>

To test how much guidence the rabbit is really providing I hand crafted some maps where exhastive searches by the turtle would be very ineffecient. This map was not part of the training set.

<VideoPlayer
    url="/blog/memorycomm/scouts_fork.mp4"
    description=""
    alt="A video of a cart-pole policy after 800,000 steps training"
/>

Novel maps don't always work, in this one the turtle nearly sees the goal before turing around.

<VideoPlayer
    url="/blog/memorycomm/scouts_fork_fail.mp4"
    description=""
    alt="A video of a cart-pole policy after 800,000 steps training"
/>

---

### Find & Return
This environment was designed to test the ability for agents to explore a unknown map for a goal location and once found to use thier memory of the space to return as quickly as possible.

Because there's only one goal location I added the ability to dig walls to no goal location will be unreachable. Digging a wall adds a timeout to the agent before it can act again to make digging walls slower and more costly

Search for goal flags in a procedurally-generated maze with destructible walls. Agents are teleported to a new random position after finding a flag — testing persistent spatial memory.


<VideoPlayer
    url="/blog/memorycomm/return_blog.mp4"
    description="A cart-pole policy after 800,000 steps training"
    alt="A video of a cart-pole policy after 800,000 steps training"
/>

Similarly to the cooperative exploration environment, agents seem to group up consistantly despite no shared reward. The simplest explination is that this helps them dig walls faster but even when wall digging is disabled the grouping behavoir presists. I designed a map (again this wasn't in the training set) to test their ability to find a goal location when it was hidden behind a costly block of walls. The agents are able to gather and determine that they should dig through to the center.

<VideoPlayer
    url="/blog/memorycomm/return_center.mp4"
    description="A cart-pole policy after 800,000 steps training"
    alt="A video of a cart-pole policy after 800,000 steps training"
/>

## Method

All of the videos shown used the same model weights trained on both the environments simultaniusly as a even split. Two other competative environments not shown here were also part of the training set. The model trained for about 4 hours on my 5090 at roughly 700k steps per second.

<Image url="/blog/memorycomm/mapox_model_architecture.svg" />

## Results

<Image url="/blog/memorycomm/rewards.png" />

<Image url="/blog/memorycomm/agent_count.png" />

### Scaling Behavior
[Depth and width scaling charts from the presentation — transformer memory benefits more from depth than width in these environments]

### Communication Scales with Agent Count
Increasing agent count improved *individual* reward — a sign of genuine emergent coordination rather than just parallelism.

---

## Hypers


### Model

| Parameter | Value |
|---|---|
| Hidden dim | 256 |
| Layers | 16 |
| Attention | GQA, 4Q / 1KV, head dim 32 |
| RoPE wavelength | 10,000 |
| FFN | 768-dim, GLU |
| Activation | GELU |
| Normalization | RMS norm, pre-norm only |
| Value head | HL-Gauss, 51 logits over [−10, 10], σ = 0.104 |
| Value hidden dim | 768 |
| CNN encoder | 3×3 kernels, strides 2/1/1, channels 16 → 32 |
| Compute dtype | bfloat16 |
| Param dtype | float32 |
| Init | Glorot uniform |

### Training

| Parameter | Value |
|---|---|
| Optimizer | Muon |
| Learning rate | 3.0 × 10⁻³ |
| Weight decay | 1.07 × 10⁻⁴ |
| AdamW β₁ / β₂ | 0.935 / 0.904 |
| Max grad norm | 0.396 |
| Algorithm | PPO |
| Minibatches | 32 |
| Discount (γ) | 0.99 |
| GAE λ | 0.91 |
| Value coeff | 0.539 |
| Value clip | 0.283 |
| Entropy coeff | 0.001 → 0.0 (annealed) |
| Advantage norm | yes |
| Max episode steps | 512 |
| Update steps | 5,000 |
