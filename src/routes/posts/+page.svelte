<script lang="ts">
	import Seo from '$lib/components/seo.svelte';
	import { formatDate } from '$lib/utils';

	let { data } = $props();
</script>

<Seo title="Posts" description="Occasional writing on my projects and ideas." />

<div class="content-column">
	<hgroup>
		<h1 class="page-title">Posts</h1>
		<p class="page-subtitle">Occasional writing on my projects and ideas.</p>
	</hgroup>
	<div class="item-list">
		{#each data.posts as post}
			<article class="post">
				<time class="meta" datetime={post.date}>{formatDate(post.date)}</time>
				<h2 class="item-title">
					<a href={'posts/' + post.slug}>{post.title}</a>
				</h2>
				<p class="item-description">{post.description}</p>
			</article>
		{/each}
	</div>
</div>

<style>
	.post {
		/* Plain list entry, not a Pico card */
		margin: 0;
		padding: 0;
		background: none;
		box-shadow: none;
	}
</style>
