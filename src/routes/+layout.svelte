<script lang="ts">
	import '@fontsource/atkinson-hyperlegible-next/400.css';
	import '@fontsource/atkinson-hyperlegible-next/400-italic.css';
	import '@fontsource/atkinson-hyperlegible-next/600.css';
	import '@fontsource/atkinson-hyperlegible-next/700.css';
	import '@fontsource/jetbrains-mono/400.css';
	import '@fontsource/jetbrains-mono/400-italic.css';

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
