<script>
	import Sun from 'lucide-svelte/icons/sun';
	import Moon from 'lucide-svelte/icons/moon';

	import { theme, toggleTheme } from '$lib/theme';
</script>

<header>
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
