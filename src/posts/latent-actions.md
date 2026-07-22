---

title: Treating LLM Latents as Continuous Actions
description: A proposal for RL fine-tuning LLMs in latent space instead of token space — making pathwise gradients (SAC-style) applicable and sidestepping the 150k-way discrete exploration problem.
date: '2026-07-21'
published: true
---

<script>
    import Image from "$lib/components/image.svelte";
    import EpisodeViewer from "$lib/components/episode/episodeViewer.svelte";
    import ArchitectureDiagram from "$lib/components/valmArchitecture.svelte";
</script>

LLM RL fine-tuning typically trains token distributions using policy gradient losses. These token vocabularies are huge—a ~150k-way discrete action space—and the optimizer knows nothing about the relationship between these tokens. This is a hard exploration problem! Also, we often don't care about the exact token selection per se; this is the wrong level of abstraction to think on. I'll argue it's possible to train an LLM at a higher level of abstraction, at an arbitrary layer of the network in latent space. Here I have an (as of now untested) idea for how to train an LLM with RL.

The core idea is to treat an intermediate latent representation inside an LLM as the action, rather than treating the sampled token as the action. A subnetwork of the model becomes the policy: its inputs are the observations (the preceding hidden state), and its outputs are continuous latent actions. Everything downstream of that latent representation—including the remainder of the network and token generation—is treated as part of the environment.

Because the action space is now continuous, pathwise policy gradient methods become applicable. Rather than only increasing the probability of sampled tokens, the critic can provide gradients that indicate which direction in latent space would improve expected return. This is analogous to continuous-control RL methods such as SAC, where gradients pass through the Q-function to optimize the policy.

An interesting consequence is that there are now two sources of stochasticity:

1. A latent-space sampler, which is part of the policy and provides exploration over continuous latent actions.
2. The existing token sampler, which remains part of the environment dynamics rather than the policy itself.

This differs from standard policy-gradient methods, where the sampled token is the action being optimized. Tokens are still sampled during rollout generation but token sampling can be completely ignored within the loss function.

This framing is related to prior work on action embeddings for discrete action spaces. Existing methods often map continuous actions to the nearest discrete embedding ("snapping"), as in [Deep Reinforcement Learning in Large Discrete Action Spaces](https://arxiv.org/abs/1512.07679) (Dulac-Arnold et al., 2015). However, this creates hard boundaries: the critic only trains on the values of discrete actions and learns nothing about the space between them.

LLMs already perform probabilistic token sampling, which suggests an alternative. If a latent representation lies between two token embeddings, the model naturally samples each token with some probability. Rather than snapping to the nearest embedding, this effectively creates a stochastic interpolation between discrete actions. In this sense, nearest-neighbor projection behaves like a hard max over embeddings, while token sampling behaves more like a softmax, producing a smoother value function over latent space.

LLMs also conveniently provide action embeddings as their token embeddings; unlike most RL work with action embeddings (e.g., [Learning Action Representations for Reinforcement Learning](https://arxiv.org/pdf/1902.00183), Chen et al., 2019), they do not need to be learned. The token sampling can be seen as a probabilistic mapping function between a latent action from a inner policy and a discrete action from the outer policy. Deterministic mapping functions create a value function that is piecewise constant within regions while probabilistic mappings replace that with a smooth mixture and a smooth landscape is exactly the precondition for pathwise gradients to exist at all. 

I plan to explore this idea with my existing [valm framework](https://github.com/gabe00122/valm) (see also [my post on valm](/posts/valm)).

Since I control the model inference infrastructure it should be possible to insert a SAC style sampler within the network during rollout generation.
