<script lang="ts">
	import ProjectCard from './projectCard.svelte';
</script>

<svelte:head>
	<title>Projects</title>
</svelte:head>

<div class="container">
	<article class="centered-card">
		<hgroup>
			<h2>Projects</h2>
		</hgroup>
		<hr />
		<div class="project-list">
			<ProjectCard
				name="Mapox"
				description="JAX-native, multi-agent, partially-observable gridworld environments with a shared observation/action format."
				repo="https://github.com/gabe00122/mapox"
			/>
			<hr />
			<ProjectCard
				name="Mapox Trainer"
				writeup="/posts/transformer_rl"
				description="High performance Jax based multi-agent ppo implementation with first class support for transformers."
				repo="https://github.com/gabe00122/mapox-trainer"
			/>
			<hr />
			<ProjectCard
				name="VALM"
				description="JAX-based framework for online RL with LLMs. Custom Qwen3 implementation with LoRA support."
				repo="https://github.com/gabe00122/valm"
			/>
			<hr />
			<ProjectCard
				name="Self Play Tic-Tac-Toe"
				description="Learning to match minmax in tictactoe with less than a minute of self play on a single GPU."
				demo="/projects/tictactoe"
				repo="https://github.com/gabe00122/tictactoe-rl"
			/>
			<hr />
			<ProjectCard
				name="Yelp Language Model"
				description="Learning exercise training a 109m parameter transformer from scratch to generate yelp reviews and predict ratings."
				repo="https://huggingface.co/gabe00122/sentiment_lm"
			/>
		</div>
	</article>
</div>
