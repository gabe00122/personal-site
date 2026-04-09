---
title: Memory and Implicit Communication
description: Emergent multi-agent signaling with spatial memory
date: '2026-04-09'
published: true
---

<script>
    import VideoPlayer from "../routes/components/video.svelte";
    import Image from "../routes/components/image.svelte";
</script>

The two demos below preview the main result of this post: agents with no explicit communication channel can still use movement and visibility as a signaling mechanism based on their in-context spacial memory.

<VideoPlayer
    url="/blog/memorycomm/return_center.mp4"
    description="Demo 1: Find & Return"
    alt="Multiple agents navigate a hand-crafted maze, digging through walls to converge on a hidden flag at the center of the map"
/>

These agents haven't been trained on this map, have separate memory and rewards and yet surprisingly quickly they seem to come to a collective decision that they should explore the center of the map even though it is more costly then exploring the outside.

<VideoPlayer
    url="/blog/memorycomm/scouts_fork.mp4"
    description="Demo 2: Cooperative Exploration"
    alt="A fast rabbit agent and a slow turtle agent navigate a map with long forked hallways. The rabbit explores ahead and positions itself so the turtle follows the correct corridor to the flag."
/>

Similarly the rabbit and turtle haven't been trained on this map and the turtle is so slow exploring all of the corridors would be costly, the rabbit is able to point the turtle down the right hallway by staying in its field of view.

---

## Spatial Memory and Communication

I started this project to explore emergent communication in multi-agent reinforcement learning, but I quickly found that memory was a necessary prerequisite. Agents need memory both to gather information worth communicating and to make use of communicated information later.

Multi-agent environments are naturally partially observable. Not only is the world state hidden, but other agents also have private observations and internal state. If another agent uses memory, then its behavior depends on what it has seen before, not just what it sees now. From your perspective, that memory is part of the hidden state.

This creates a recursive problem: memory helps an agent deal with partial observability, but once one agent has memory, every other agent faces a harder inference problem. They now have to model policies conditioned on latent internal state.

That is why I became interested in memory before communication. In this post, I use "communication" broadly: not just explicit messages, but any observable behavior that changes another agent's future beliefs or decisions.

Communication and exploration also have a similar structure. Exploration improves your own future decisions by improving your knowledge of the world. Communication improves another agent's future decisions by changing what they know about the world. If agents have different goals, that change does not have to be helpful. Honest signaling, selective disclosure, and deception are all forms of communication.

---

## What I Built

I wanted to see if model-free RL with a transformer over time could learn something similar to spatial memory and if this spatial memory could be communicated between agents with sufficient self-play. I also wanted to see if training the same model on multiple tasks would lead to task-to-task skill transfer.

This builds on the transformer RL framework described in my [previous post](/posts/transformer_rl). The training framework and can be found at [jaxrl](https://github.com/gabe00122/jaxrl) and the environments [mapox](https://github.com/gabe00122/mapox)

The checkpoint shown here was trained jointly on four environments. I focus on two of them because they most clearly expose the role of memory and implicit communication. The other two: a competitive king-of-the-hill game and a predator-prey game-were part of the multitask training mix but deserve separate analysis because the incentives and learned behaviors are qualitatively different.

---

### Cooperative Exploration

To test if agents could share information from their memory when incentives were perfectly aligned I designed a simple asymmetric exploration game

There are two agent types, a **fast rabbit** and a **slow turtle** agent. Both agents are rewarded when the **turtle reaches the flag** location.

- Both agents have separate memory
- Agents have an egocentric field of view highlighted by the shaded square
- Each episode uses a new map from a map generator
- The rabbit's only effect on the environment is to observe and to be observed by the turtle
- There is no explicit communication channel

Here's an example episode on the type of map these agents are trained on:
<VideoPlayer
    url="/blog/memorycomm/scouts.mp4"
    description="Typical training map, the map is generated using perlin noise and there are few strait hallways"
    alt="The rabbit quickly sweeps the map while the turtle follows at a slower pace, eventually reaching the flags."
/>

To test generalization I hand-crafted maps to challenge the agents, this is what was used for the first two video demonstrations, the challenge maps are never in the training set. The scouts don't get distracted by unreachable goal locations. They seem to know the difference between one they can reach and one they can't get to.

<VideoPlayer
  url="/blog/memorycomm/scouts_blog_d.mp4"
  description="Hand-crafted map: agents ignore a visible but unreachable flag behind a wall"
  alt="Agents navigate past a flag visible through a wall but blocked by it. Rather than attempting to reach the inaccessible flag, they continue exploring and find the reachable goal."
/>

Hand-crafted maps don't always work, in this one the turtle nearly sees the goal before turning around. I think something about the corridors being close together confused them. Perhaps the scout developed an exploration heuristic where it prefers a certain spacing to get good coverage of the map but this confuses the communication signal between agents.

<VideoPlayer
    url="/blog/memorycomm/scouts_fork_fail.mp4"
    description="Failure case: tightly packed corridors confuse the rabbit's exploration pattern, and the turtle turns back just before seeing the goal"
    alt="A map with many narrow hallways packed close together on the left side. The rabbit navigates the corridors but the turtle agent turns around one tile short of seeing the flag — a failure of the implicit communication signal."
/>

Training on the other environments leads to much better performance than training on the scout environment alone. My theory is the relationship between the scout and the turtle is inherently unstable because they're both adapting to each other at the same time so it's a challenging moving target. The other environments help with a core spatial memory ability that benefits this environment but is challenging to learn from this environment alone.


<Image url="/blog/memorycomm/scouts_comparison.webp" description="Single-task vs. multitask training reward curves for the cooperative exploration environment" alt="Line graph comparing scouts trained single-task (blue) versus multitask (red) over 5000 steps. The multitask run reaches roughly 5.0 reward while the single-task run plateaus around 3.5–4.0, a ~20–30% gap. Both have wide confidence bands indicating run-to-run variability." />

I wondered if scaling up the proportion of the population that were scouts would improve the average reward but this does not seem to be the case. Having one rabbit and one turtle per environment seems to be a sweet spot. I thought that 3 rabbits per turtle would lead to higher average reward but it seems to be lower than just one to one. A turtle with no rabbits performs worse than with one rabbit but better than the one to one ratio without multitask training. With optimal training more rabits should lead to strictly better exploration and therefor reward, the fact that it does not suggests a shortcoming in my current training approach.

<Image url="/blog/memorycomm/scouts_reward.webp" description="Effect of rabbit count on average reward: one rabbit outperforms zero or three" alt="Line graph comparing reward over 5000 training steps for zero scouts (blue, ~4.4), one scout (red, ~5.1), and three scouts (green, ~4.8). One scout achieves the best final reward; adding more scouts beyond one does not help and slightly hurts performance." />

---

### Find & Return
Initially this environment was designed to test a single agent's spatial memory but interesting things happen when you add multiple agents into the environment together.

- Agents spawn in a random location on a randomized map and must find the flag location
- When agents find the flag they are given a reward and teleported to a random new location, this is still a single episode
- Walls can be destroyed but causes a timeout to the agent that destroyed it
- Agents can see each other but otherwise don't interact directly

<VideoPlayer
    url="/blog/memorycomm/return_blog.mp4"
    description="Typical training map: agents spawn at random positions and search for the flag"
    alt="Several agents spawn at random positions on a procedurally generated maze. They explore independently, find the flag, collect the reward, and are teleported to new random locations to repeat the process within the same episode."
/>

Agents seem to group up consistently despite no shared reward. The simplest explanation is that this helps them dig walls faster but even when wall digging is disabled the grouping behavior persists.

<VideoPlayer
  url="/blog/memorycomm/leader_slow.mp4"
  description="Manual control demo: a human-controlled agent leads the others — they follow when it changes direction or starts digging"
  alt="A human-controlled agent navigates the maze while nearby autonomous agents mirror its direction changes and digging actions, showing that agents use observed movement as an implicit signal."
/>

You can lead the agents around if you manually control one, they don't always follow you but if you act decisively it's more likely that they will. Sometimes the agent in the rear of the group initiates the group to change course.

<Image url="/blog/memorycomm/agent_count.webp" description="Episode reward scales reliably with agent count in the Find & Return environment" alt="Line graph showing episode reward over 1000 training steps for 4 agents (purple, ~14.2), 8 agents (teal, ~15.1), and 16 agents (yellow, ~16.1). More agents consistently yields higher reward, with all configurations showing rapid learning in the first 200 steps then gradual improvement." />

Contrary to the cooperative exploration environment, in this environment individual reward scales up reliably with agent count. This strongly suggests they are gaining a benefit from being in groups and the benefit goes up as the groups become larger.

---

### Conclusion

This work barely scratches the surface, it shows that memory and self-play alone is sufficient for some basic agent to agent signaling but particularly in the cooperative exploration environment I think there's a much higher ceiling for how well agents could coordinate. I believe there's a strong possibility that more specialized multiagent techniques would lead to much better performance and I would invite anyone interested to use these environments as a benchmark and try to improve on these results.

A natural follow-up is to study trust and deception directly. I want to build environments where agents have private, role-dependent goals: some aligned with the group, others opposed to it. In that setting, communication is no longer just about sharing information, but about judging reliability and hidden intent.
