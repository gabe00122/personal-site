<script lang="ts">
	import '@picocss/pico/css/pico.pumpkin.min.css';

	import Header from './header.svelte';
	interface Props {
		children?: import('svelte').Snippet;
	}

	let { children }: Props = $props();

	const children_render = $derived(children);
</script>

<div class="container">
	<Header />

	<main>
		{@render children_render?.()}
	</main>
</div>
