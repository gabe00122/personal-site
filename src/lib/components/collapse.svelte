<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		title: string;
		open?: boolean;
		children: Snippet;
	}

	let { title, open = false, children }: Props = $props();
</script>

<details class="collapse" {open}>
	<summary>{title}</summary>
	<div class="collapse-content">
		{@render children()}
	</div>
</details>

<style>
	.collapse {
		border: 1px solid var(--pico-muted-border-color);
		border-radius: var(--pico-border-radius);
		padding: 0;
		margin-bottom: var(--pico-typography-spacing-vertical);
	}

	.collapse > summary {
		padding: calc(var(--pico-spacing) * 0.75) var(--pico-spacing);
		margin: 0;
	}

	.collapse[open] > summary {
		border-bottom: 1px solid var(--pico-muted-border-color);
	}

	.collapse-content {
		padding: var(--pico-spacing);
	}

	.collapse-content > :global(:last-child) {
		margin-bottom: 0;
	}
</style>
