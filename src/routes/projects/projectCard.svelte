<script lang="ts">
	interface Props {
		name: string;
		description: string;
		demo?: string;
		repo: string;
		writeup?: string;
	}

	const { name, description, demo, repo, writeup }: Props = $props();
</script>

<section class="project">
	<header class="pc-header">
		<h2 class="item-title"><a href={repo}>{name}</a></h2>
		<span class="pc-links meta">
			{#if demo}
				<a href={demo}>demo</a>
			{/if}
			{#if writeup}
				<a href={writeup}>writeup</a>
			{/if}
		</span>
	</header>
	<p class="item-description">{description}</p>
</section>

<style>
	.project {
		margin: 0;
	}
	.pc-header {
		display: flex;
		align-items: baseline;
		gap: 1rem;
	}
	.pc-header .item-title {
		margin-right: auto;
	}
	.pc-links {
		display: flex;
		gap: 1rem;
		white-space: nowrap;
	}
</style>
