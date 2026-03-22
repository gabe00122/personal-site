<script lang="ts">
	import ProjectCard from './projectCard.svelte';
</script>

<svelte:head>
	<title>Projects</title>
</svelte:head>

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
			name="JaxRL"
			writeup="/posts/transformer_rl"
			description="High preformace Jax based multi-agent ppo implementation with first class support for transformers."
			repo="https://github.com/gabe00122/jaxrl"
		/>
		<hr />
		<ProjectCard
			name="LlmRL"
			description="Multi-step RL LLM fine tuning build from the ground up in Jax."
			repo="https://github.com/gabe00122/llm_rl"
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
