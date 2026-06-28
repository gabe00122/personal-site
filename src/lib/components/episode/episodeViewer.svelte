<script lang="ts">
	import { untrack } from 'svelte';
	import { decodeEpisode } from './decode';
	import type { Episode } from './types';
	import Tokens from './tokens.svelte';
	import Graph from './graph.svelte';
	import TokenDetail from './tokenDetail.svelte';
	import { metricOptionLabel } from './metricFormat';
	import { firstUnmaskedPolicyToken, getPolicyTokenMask, isMaskedPolicyToken } from './policyMask';

	interface EpisodeTab {
		/** URL of a static episode JSON file (EncodedEpisode shape). */
		url: string;
		/** Label shown on the tab. */
		label: string;
	}

	interface Props {
		/** URL of a single static episode JSON file (EncodedEpisode shape). */
		url?: string;
		/** Label for the single-episode tab. */
		title?: string;
		/** Multiple episodes, each shown as a clickable tab. */
		episodes?: EpisodeTab[];
		/** Metric used for the initial token heatmap + graph. */
		metric?: string;
		/** Height of the scrollable token area. */
		tokensHeight?: string;
	}

	let { url, title, episodes, metric = 'rewards', tokensHeight = '18rem' }: Props = $props();

	let tabs = $derived<EpisodeTab[]>(
		episodes && episodes.length > 0 ? episodes : url ? [{ url, label: title ?? '' }] : []
	);
	let showTabs = $derived(tabs.length > 1 || (tabs.length === 1 && tabs[0].label !== ''));

	let activeTab = $state(0);
	let episode = $state<Episode | null>(null);
	let loadError = $state(false);
	let metricKey = $state('none');
	let selectedIndex = $state<number | null>(null);
	let hoveredIndex = $state<number | null>(null);
	// Tracks whether any episode has loaded yet; the first one defaults to the `metric` prop.
	let metricInitialized = false;

	let metricOptions = $derived(episode ? ['none', ...Object.keys(episode.tokenMetrics)] : ['none']);
	let policyTokenMask = $derived(episode ? getPolicyTokenMask(episode, metricKey) : null);

	const cache = new Map<string, Episode>();
	let loadToken = 0;

	function resolveMetric(decoded: Episode, preferred: string): string {
		if (preferred === 'none' || preferred in decoded.tokenMetrics) {
			return preferred;
		}
		if (metric in decoded.tokenMetrics) {
			return metric;
		}
		return 'none';
	}

	function applyEpisode(decoded: Episode) {
		const preferred = metricInitialized ? metricKey : metric;
		metricInitialized = true;
		metricKey = resolveMetric(decoded, preferred);
		selectedIndex = 0;
		hoveredIndex = null;
		episode = decoded;
	}

	async function loadEpisode(tab: EpisodeTab) {
		const token = ++loadToken;

		const cached = cache.get(tab.url);
		if (cached) {
			loadError = false;
			applyEpisode(cached);
			return;
		}

		episode = null;
		loadError = false;

		try {
			const response = await fetch(tab.url);
			if (!response.ok) {
				throw new Error(`Failed to load episode: ${response.status}`);
			}
			const decoded = decodeEpisode(await response.json());
			cache.set(tab.url, decoded);
			if (token !== loadToken) {
				return;
			}
			applyEpisode(decoded);
		} catch (error) {
			if (token !== loadToken) {
				return;
			}
			console.error(error);
			loadError = true;
		}
	}

	// Load the active tab's episode whenever the selection (or the tab list) changes.
	$effect(() => {
		const tab = tabs[activeTab];
		if (!tab) {
			return;
		}
		untrack(() => loadEpisode(tab));
	});

	$effect(() => {
		if (episode === null) {
			return;
		}

		const tokenCount = episode.tokens.length;
		if (selectedIndex !== null && (selectedIndex < 0 || selectedIndex >= tokenCount)) {
			selectedIndex = null;
		}
		if (hoveredIndex !== null && (hoveredIndex < 0 || hoveredIndex >= tokenCount)) {
			hoveredIndex = null;
		}

		if (policyTokenMask === null) {
			return;
		}

		if (selectedIndex !== null && isMaskedPolicyToken(policyTokenMask, selectedIndex)) {
			selectedIndex = firstUnmaskedPolicyToken(policyTokenMask, tokenCount);
		}
		if (hoveredIndex !== null && isMaskedPolicyToken(policyTokenMask, hoveredIndex)) {
			hoveredIndex = null;
		}
	});
</script>

<figure class="episode-viewer">
	<article class="episode-card">
		{#if showTabs}
			<div class="episode-tabs" role="tablist">
				{#each tabs as tab, index (tab.url)}
					<button
						type="button"
						role="tab"
						class="episode-tab"
						class:selected={index === activeTab}
						aria-selected={index === activeTab}
						onclick={() => (activeTab = index)}
					>
						{tab.label || `Episode ${index + 1}`}
					</button>
				{/each}
			</div>
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

	.episode-tabs {
		display: flex;
		overflow-x: auto;
		border-bottom: 1px solid var(--pico-border-color);
	}

	.episode-tab {
		/* Pico styles a bare <button> as a filled primary button, which locally
		   redefines --pico-color (to the inverse text color, #000 in dark mode).
		   Reset it so var(--pico-color) below resolves to the themed page text. */
		--pico-color: inherit;
		--pico-background-color: transparent;
		box-shadow: none;
		flex: 0 0 auto;
		width: auto;
		margin: 0;
		padding: 0.5rem 0.9rem;
		border: none;
		border-bottom: 2px solid transparent;
		border-radius: 0;
		background: transparent;
		color: var(--pico-muted-color);
		font-size: 0.8rem;
		font-weight: 500;
		white-space: nowrap;
		cursor: pointer;
		transition:
			color 0.15s ease,
			border-color 0.15s ease,
			background 0.15s ease;
	}

	.episode-tab:hover {
		color: var(--pico-color);
		background: color-mix(in srgb, var(--pico-color) 7%, transparent);
	}

	.episode-tab.selected {
		color: var(--pico-primary);
		border-bottom-color: var(--pico-primary);
		background: color-mix(in srgb, var(--pico-primary) 10%, transparent);
	}

	.episode-tab.selected:hover {
		background: color-mix(in srgb, var(--pico-primary) 16%, transparent);
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
