<script lang="ts">
	import '@picocss/pico/css/pico.pumpkin.min.css';
	import '../app.css';

	import Header from './header.svelte';
	interface Props {
		children?: import('svelte').Snippet;
	}

	let { children }: Props = $props();
</script>

<div>
	<Header />

	<main>
		{@render children?.()}
	</main>
</div>
