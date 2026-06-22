<script lang="ts">
	import { onMount } from 'svelte';
	import { decodeEpisode } from './decode';
	import type { Episode } from './types';
	import Tokens from './tokens.svelte';
	import Graph from './graph.svelte';
	import TokenDetail from './tokenDetail.svelte';
	import { metricOptionLabel } from './metricFormat';

	interface Props {
		/** URL of a static episode JSON file (EncodedEpisode shape). */
		url: string;
		/** Optional title shown in the card header. */
		title?: string;
		/** Metric used for the initial token heatmap + graph. */
		metric?: string;
		/** Height of the scrollable token area. */
		tokensHeight?: string;
	}

	let { url, title, metric = 'rewards', tokensHeight = '18rem' }: Props = $props();

	let episode = $state<Episode | null>(null);
	let loadError = $state(false);
	let metricKey = $state('none');
	let selectedIndex = $state<number | null>(null);
	let hoveredIndex = $state<number | null>(null);

	let metricOptions = $derived(episode ? ['none', ...Object.keys(episode.tokenMetrics)] : ['none']);

	onMount(async () => {
		try {
			const response = await fetch(url);
			if (!response.ok) {
				throw new Error(`Failed to load episode: ${response.status}`);
			}
			const decoded = decodeEpisode(await response.json());
			metricKey = metric in decoded.tokenMetrics ? metric : 'none';
			selectedIndex = 0;
			episode = decoded;
		} catch (error) {
			console.error(error);
			loadError = true;
		}
	});
</script>

<figure class="episode-viewer">
	<article class="episode-card">
		{#if title}
			<header class="episode-title">{title}</header>
		{/if}
		{#if loadError}
			<div class="state">Could not load episode.</div>
		{:else if episode === null}
			<div class="state">Loading episode…</div>
		{:else}
			<div class="toolbar">
				<label>
					Show
					<select bind:value={metricKey}>
						{#each metricOptions as option (option)}
							<option value={option}>{metricOptionLabel(option)}</option>
						{/each}
					</select>
				</label>
				<TokenDetail {episode} {metricKey} {selectedIndex} {hoveredIndex} />
			</div>
			<div class="graph-pane">
				<Graph {episode} {metricKey} bind:selectedIndex bind:hoveredIndex />
			</div>
			<Tokens {episode} {metricKey} {tokensHeight} bind:selectedIndex bind:hoveredIndex />
		{/if}
	</article>
</figure>

<style>
	.episode-viewer {
		margin-block: var(--pico-block-spacing-vertical, 1rem);
	}

	.episode-card {
		margin: 0;
		padding: 0;
		overflow: hidden;
	}

	.episode-title {
		margin: 0;
		border-bottom: 1px solid var(--pico-border-color);
	}

	.toolbar {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		overflow-x: auto;
		padding: 0.5rem 0.75rem;
		border-bottom: 1px solid var(--pico-border-color);
		font-size: 0.8rem;
	}

	.toolbar label {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin: 0;
		color: var(--pico-muted-color);
	}

	.toolbar select {
		width: auto;
		margin: 0;
		padding: 0.15rem 1.75rem 0.15rem 0.5rem;
		font-size: 0.8rem;
		height: auto;
	}

	.graph-pane {
		height: 7rem;
		padding: 0.25rem 0.5rem;
		border-bottom: 1px solid var(--pico-muted-border-color, var(--pico-border-color));
	}

	.state {
		padding: 2rem;
		text-align: center;
		color: var(--pico-muted-color);
	}
</style>
