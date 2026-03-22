<script>
	import Sun from 'lucide-svelte/icons/sun';
	import Moon from 'lucide-svelte/icons/moon';

	import { theme, toggleTheme } from '$lib/theme';
	import { onMount } from 'svelte';

	let scrolled = $state(false);

	onMount(() => {
		const onScroll = () => {
			scrolled = window.scrollY > 0;
		};
		window.addEventListener('scroll', onScroll, { passive: true });
		return () => window.removeEventListener('scroll', onScroll);
	});
</script>

<header class:scrolled>
	<div class="container">
		<nav aria-label="Main" lang="en">
			<!-- <ul>
    			<li><a class="title contrast" href="/">Gabriel Keith</a></li>
    		</ul> -->
			<ul class="header-items">
				<li class="title"><a class="contrast" href="/">Gabriel Keith</a></li>
				<li><a class="contrast" lang="en" href="/about">About</a></li>
				<li><a class="contrast" lang="en" href="/posts">Posts</a></li>
				<!-- <li><a class="contrast" lang="en" href="/currently">Currently</a></li> -->
				<li><a class="contrast" lang="en" href="/projects">Projects</a></li>
				<!-- <li><a class="contrast" lang="en" href="/contact">Contact</a></li> -->
			</ul>
			<button onclick={toggleTheme} class="toggle-dark-button outline contrast">
				{#if $theme === 'dark'}
					<Sun />
				{:else}
					<Moon />
				{/if}
			</button>
		</nav>
	</div>
</header>

<style>
	.scrolled {
		top: 0;
		position: sticky;
		backdrop-filter: blur(1rem);
		z-index: 2;

		background-color: var(--pico-header-background);
		box-shadow: var(--pico-card-box-shadow);
		border-bottom: var(--pico-border-width) solid transparent;
		border-bottom-color: var(--pico-header-border-color);

		transition:
			border-top-color 0.4s ease-in-out,
			box-shadow 0.4s ease-in-out;
	}

	header .container {
		padding-right: 0;
	}

	.header-items {
		width: 100%;
		display: flex;
	}

	.title {
		margin-right: auto;
	}

	.title a {
		font-weight: bold;
	}

	.toggle-dark-button {
		border: none;
	}
</style>
