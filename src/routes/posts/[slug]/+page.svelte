<script lang="ts">
	import { formatDate } from '$lib/utils';

	let { data } = $props();
</script>

<svelte:head>
	<title>{data.meta.title}</title>
	<meta property="og:type" content="article" />
	<meta property="og:title" content={data.meta.title} />
</svelte:head>

<div class="container post-card">
	<article>
		<hgroup>
			<h1>{data.meta.title}</h1>
			<p>Published at {formatDate(data.meta.date)}</p>
		</hgroup>

		<!-- <div class="tags">
    		{#each data.meta.categories as category}
    			<span class="surface-4">&num;{category}</span>
    		{/each}
    	</div> -->

		<div>
			<data.content />
		</div>
	</article>
</div>

<style>
	.post-card {
		margin-top: 1rem;
	}
</style>
