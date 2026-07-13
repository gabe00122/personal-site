<script lang="ts">
	import { formatDate } from '$lib/utils';
	import Seo from '$lib/components/seo.svelte';

	let { data } = $props();
</script>

<Seo title={data.meta.title} description={data.meta.description} type="article" />

<div class="container">
	<article class="centered-card">
		<hgroup>
			<h1>{data.meta.title}</h1>
			<p>Published at {formatDate(data.meta.date)}</p>
		</hgroup>

		<div>
			<data.content />
		</div>
	</article>
</div>
