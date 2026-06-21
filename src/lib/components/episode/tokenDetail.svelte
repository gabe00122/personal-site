<script lang="ts">
	import type { Episode } from './types';

	interface Props {
		episode: Episode;
		metricKey: string;
		selectedIndex: number | null;
		hoveredIndex: number | null;
	}

	let { episode, metricKey, selectedIndex, hoveredIndex }: Props = $props();

	let activeIndex = $derived(hoveredIndex ?? selectedIndex);
	let activeMetricValue = $derived(getMetricValue(activeIndex));

	function getMetricValue(index: number | null) {
		if (index === null || metricKey === 'none') {
			return null;
		}

		return episode.tokenMetrics[metricKey]?.[index] ?? null;
	}

	function formatMetricValue(value: number | null) {
		if (value === null || !Number.isFinite(value)) {
			return 'n/a';
		}

		return value.toLocaleString(undefined, {
			maximumSignificantDigits: 6
		});
	}
</script>

<div class="token-detail">
	{#if activeIndex === null}
		<span>
			<span class="label">Index</span>
			<span class="value muted">n/a</span>
		</span>
		<span>
			<span class="label">{metricKey === 'none' ? 'Metric' : metricKey}</span>
			<span class="value muted">n/a</span>
		</span>
	{:else}
		<span>
			<span class="label">Index</span>
			<span class="value">{activeIndex}</span>
		</span>
		<span>
			<span class="label">{metricKey === 'none' ? 'Metric' : metricKey}</span>
			<span class="value">{formatMetricValue(activeMetricValue)}</span>
		</span>
	{/if}
</div>

<style>
	.token-detail {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		min-width: 0;
		font-size: 0.75rem;
		white-space: nowrap;
	}

	.label {
		color: var(--pico-muted-color);
		margin-right: 0.3rem;
	}

	.value {
		font-family: var(--pico-font-family-monospace);
		color: var(--pico-color);
	}

	.muted {
		color: var(--pico-muted-color);
	}
</style>
