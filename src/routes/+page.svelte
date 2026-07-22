<script lang="ts">
	import Seo from '$lib/components/seo.svelte';
	import { onMount } from 'svelte';

	let video: HTMLVideoElement | undefined = $state();

	onMount(() => {
		if (!video) return;
		if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
			video.pause();
		} else {
			// Half speed: ambient, and the loop seam comes around half as often
			video.playbackRate = 0.5;
		}
	});
</script>

<Seo description="I teach agents from experience in social environments." />

<div class="hero">
	<!-- Decorative gridworld from the memory & communication experiments;
	     swap for the live WASM sim when it's ready -->
	<video
		bind:this={video}
		src="/hero/return_loop.mp4"
		autoplay
		muted
		loop
		playsinline
		aria-hidden="true"
	></video>
	<div class="landing">
		<h1>Gabriel Keith</h1>
		<p>I teach agents from experience in social environments.</p>
	</div>
</div>

<style>
	.hero {
		position: relative;
		min-height: calc(100svh - 10rem);
		display: flex;
		align-items: center;
		justify-content: center;
		overflow: hidden;
	}

	.hero video {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		object-fit: cover;
		opacity: 0.24;
		pointer-events: none;
		-webkit-mask-image: radial-gradient(ellipse 70% 60% at 50% 45%, black 30%, transparent 75%);
		mask-image: radial-gradient(ellipse 70% 60% at 50% 45%, black 30%, transparent 75%);
	}

	/* Decorative motion: respect reduced-motion, and the dark footage
	   has no good light-mode treatment yet */
	@media (prefers-reduced-motion: reduce) {
		.hero video {
			display: none;
		}
	}

	:global(html[data-theme='light']) .hero video {
		display: none;
	}

	.landing {
		position: relative;
		text-align: center;
		padding: 0 var(--pico-spacing) 8vh;
	}

	.landing h1 {
		--pico-font-size: 3rem;
		font-size: 3rem;
		margin-bottom: 0.25rem;
	}

	.landing p {
		color: var(--pico-muted-color);
	}
</style>
