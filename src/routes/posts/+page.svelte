<script lang="ts">
	import { formatDate } from '$lib/utils.js';
	let { data } = $props();
</script>

<div class="container">
	<article class="centered-card">
		<hgroup>
			<h2 class="header-margin">Posts</h2>
			<p>Occasional writing on my projects and ideas.</p>
		</hgroup>
		<div class="posts">
			{#each data.posts as post}
				<div class="post">
					<hr />
					<a href={'posts/' + post.slug}>
						<strong>{post.title}</strong>
					</a>
					<br />
					<small>
						{post.description}
					</small>
				</div>
			{/each}
		</div>
	</article>
</div>

<style>
	.header-margin {
		margin-bottom: 0.75rem;
	}
	.posts {
		display: flex;
		flex-direction: column;
	}
</style>
