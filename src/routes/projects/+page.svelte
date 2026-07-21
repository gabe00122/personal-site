<script lang="ts">
	import ProjectCard from './projectCard.svelte';
	import Seo from '$lib/components/seo.svelte';
</script>

<Seo
	title="Projects"
	description="Projects in reinforcement learning, JAX, and multi-agent systems."
/>

<div class="content-column">
	<hgroup>
		<h1 class="page-title">Projects</h1>
		<p class="page-subtitle">Things I've built, mostly reinforcement learning in JAX.</p>
	</hgroup>
	<div class="item-list">
		<ProjectCard
			name="Mapox"
			description="JAX-native, multi-agent, partially-observable gridworld environments with a shared observation/action format."
			repo="https://github.com/gabe00122/mapox"
		/>
		<ProjectCard
			name="Mapox Trainer"
			writeup="/posts/transformer_rl"
			description="High performance Jax based multi-agent ppo implementation with first class support for transformers."
			repo="https://github.com/gabe00122/mapox-trainer"
		/>
		<ProjectCard
			name="VALM"
			writeup="/posts/valm"
			description="JAX-based framework for online RL with LLMs. Custom Qwen3 implementation with LoRA support."
			repo="https://github.com/gabe00122/valm"
		/>
		<ProjectCard
			name="Self Play Tic-Tac-Toe"
			description="Learning to match minmax in tictactoe with less than a minute of self play on a single GPU."
			demo="/projects/tictactoe"
			repo="https://github.com/gabe00122/tictactoe-rl"
		/>
		<ProjectCard
			name="Yelp Language Model"
			description="Learning exercise training a 109m parameter transformer from scratch to generate yelp reviews and predict ratings."
			repo="https://huggingface.co/gabe00122/sentiment_lm"
		/>
	</div>
</div>
