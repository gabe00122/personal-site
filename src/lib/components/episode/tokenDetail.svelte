<script lang="ts">
	import type { Episode } from './types';
	import { metricDetailLabel } from './metricFormat';
	import { getPolicyTokenMask, isMaskedPolicyToken } from './policyMask';

	interface Props {
		episode: Episode;
		metricKey: string;
		selectedIndex: number | null;
		hoveredIndex: number | null;
	}

	let { episode, metricKey, selectedIndex, hoveredIndex }: Props = $props();

	let policyTokenMask = $derived(getPolicyTokenMask(episode, metricKey));
	let activeIndex = $derived(getActiveIndex(hoveredIndex ?? selectedIndex));
	let activeMetricValue = $derived(getMetricValue(activeIndex));
	let activeMetricLabel = $derived(metricDetailLabel(metricKey));
	let activeMetricDisplay = $derived(
		activeIndex === null ? 'n/a' : formatMetricValue(activeMetricValue)
	);

	function getActiveIndex(index: number | null) {
		if (index === null || isMaskedPolicyToken(policyTokenMask, index)) {
			return null;
		}

		return index;
	}

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
	<span>
		<span class="label">Index</span>
		<span class="value" class:muted={activeIndex === null}>{activeIndex ?? 'n/a'}</span>
	</span>
	<span>
		<span class="label">{activeMetricLabel}</span>
		<span class="value" class:muted={activeIndex === null}>{activeMetricDisplay}</span>
	</span>
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

	.value.muted {
		color: var(--pico-muted-color);
	}
</style>
