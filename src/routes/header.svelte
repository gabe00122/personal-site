<script>
	import Sun from 'lucide-svelte/icons/sun';
	import Moon from 'lucide-svelte/icons/moon';

	import { theme, toggleTheme } from '$lib/theme';
</script>

<header>
	<nav aria-label="Main" lang="en">
		<ul>
			<li><strong>Gabriel Keith</strong></li>
			<li>
				<button onclick={toggleTheme} class="outline contrast" style="border: none;">
					{#if $theme === 'dark'}
						<Sun />
					{:else}
						<Moon />
					{/if}
				</button>
			</li>
		</ul>
		<ul>
			<li><a lang="en" href="/">About</a></li>
			<li><a lang="en" href="/blog">Blog</a></li>
			<li><a lang="en" href="/projects">Projects</a></li>
			<!-- <li><a lang="en" href="/music">Music</a></li> -->
		</ul>
	</nav>
</header>
