<script lang="ts">
	import { formatDate } from '$lib/utils.js';
	let { data } = $props();
</script>

<div class="container">
	<article class="centered-card">
		<hgroup>
			<h2 class="header-margin">Posts</h2>
			<p>
				Occasional writing on my projects and ideas. Some of these are long-form technical posts,
				some are quick observations.
			</p>
		</hgroup>
		<hr />
		<div class="posts">
			{#each data.posts as post}
				<div class="post">
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
	.post {
		margin-bottom: 1rem;
	}
</style>
