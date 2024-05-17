<script lang="ts">
	import '@picocss/pico/css/pico.pumpkin.min.css';

	import Header from './header.svelte';
</script>

<div class="container">
	<Header />

	<main>
		<slot />
	</main>
</div>
