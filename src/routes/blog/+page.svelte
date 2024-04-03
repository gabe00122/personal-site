<script lang="ts">
	import * as config from '$lib/config'
    import { formatDate } from "$lib/utils.js";

	export let data
</script>


<!-- Posts -->
<section>
	<ul class="posts">
		{#each data.posts as post}
			<li class="post">
				<a href={"blog/" + post.slug} class="title">{post.title}</a>
				<p class="date">{formatDate(post.date)}</p>
				<p class="description">{post.description}</p>
			</li>
		{/each}
	</ul>
</section>