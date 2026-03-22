<script lang="ts">
	interface Props {
		name: string;
		description: string;
		demo?: string;
		repo: string;
		writeup?: string;
		thumbnail?: string;
	}

	const { name, description, demo, repo, writeup }: Props = $props();
</script>

<section>
	<!-- {#if thumbnail}
		<img src={thumbnail} alt="{name} thumbnail" />
	{/if} -->
	<header class="pc-header">
		<strong class="pc-title"><a href={repo}>{name}</a></strong>
		{#if demo}
			<a href={demo}>Demo</a>
		{/if}
		{#if writeup}
			<a href={writeup}>Writeup</a>
		{/if}
	</header>
	<small>{description}</small>
	<!-- <footer> -->
	<!-- </footer> -->
</section>

<style>
	.pc-header {
		display: flex;
	}
	.pc-title {
		margin-right: auto;
	}
</style>
