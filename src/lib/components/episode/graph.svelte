<script lang="ts">
	import type { Episode } from './types';
	import { Highlight, LineChart, type ChartState } from 'layerchart';
	import { metricDetailLabel } from './metricFormat';
	import { getPolicyTokenMask, isMaskedPolicyToken } from './policyMask';

	interface Props {
		episode: Episode;
		metricKey: string;
		selectedIndex: number | null;
		hoveredIndex: number | null;
	}

	type MetricDatum = {
		index: number;
		value: number | null;
		masked: boolean;
	};

	type MaskedRange = {
		start: number;
		end: number;
	};

	const Y_AXIS_TICK_COUNT = 4;

	let {
		episode,
		metricKey,
		selectedIndex = $bindable(null),
		hoveredIndex = $bindable(null)
	}: Props = $props();

	let context = $state<ChartState>(null!);
	let metricValues = $derived(
		metricKey === 'none' ? null : (episode.tokenMetrics[metricKey] ?? null)
	);
	let policyTokenMask = $derived(getPolicyTokenMask(episode, metricKey));
	let chartData = $derived(getChartData(metricValues, policyTokenMask));
	let maskedRanges = $derived(getMaskedRanges(chartData));
	let maskedAnnotations = $derived(
		maskedRanges.map(({ start, end }) => ({
			type: 'range' as const,
			layer: 'below' as const,
			x: [start - 0.5, end + 0.5],
			class: 'metric-masked-range'
		}))
	);
	let yDomain = $derived(getYDomain(chartData));
	let hoveredDatum = $derived(getDatum(hoveredIndex));
	let selectedDatum = $derived(getDatum(selectedIndex));
	let series = $derived([
		{
			key: 'value',
			label: metricDetailLabel(metricKey),
			color: 'var(--pico-primary)'
		}
	]);

	function getChartData(
		values: ArrayLike<number> | null,
		policyTokenMask: ArrayLike<number> | null
	): MetricDatum[] {
		if (values === null) {
			return [];
		}

		const out: MetricDatum[] = [];
		for (let i = 0; i < values.length; i++) {
			const masked = isMaskedPolicyToken(policyTokenMask, i);
			const value = values[i];
			out.push({ index: i, value: masked || !Number.isFinite(value) ? null : value, masked });
		}
		return out;
	}

	function getMaskedRanges(data: MetricDatum[]): MaskedRange[] {
		const ranges: MaskedRange[] = [];
		let start: number | null = null;

		for (let i = 0; i < data.length; i++) {
			if (data[i].masked) {
				start ??= data[i].index;
			} else if (start !== null) {
				ranges.push({ start, end: data[i - 1].index });
				start = null;
			}
		}

		if (start !== null && data.length > 0) {
			ranges.push({ start, end: data[data.length - 1].index });
		}

		return ranges;
	}

	function getDatum(index: number | null) {
		if (index === null) {
			return undefined;
		}

		const datum = chartData[index];
		return datum && !datum.masked ? datum : undefined;
	}

	function getYDomain(data: MetricDatum[]): [number, number] | undefined {
		let min = Infinity;
		let max = -Infinity;
		for (const { value } of data) {
			if (value === null || !Number.isFinite(value)) {
				continue;
			}
			min = Math.min(min, value);
			max = Math.max(max, value);
		}

		if (min === Infinity) {
			return undefined;
		}

		const span = max - min;
		const padding = span === 0 ? 1 : span * 0.08;
		return [min - padding, max + padding];
	}

	function tooltipIndex() {
		const index = (context?.tooltipState.data as Partial<MetricDatum> | null)?.index ?? null;
		if (typeof index !== 'number' || chartData[index]?.masked) {
			return null;
		}
		return index;
	}

	function updateHoverIndex() {
		hoveredIndex = tooltipIndex();
	}

	function selectDatum() {
		const index = tooltipIndex();
		if (typeof index !== 'number') {
			return;
		}
		selectedIndex = selectedIndex === index ? null : index;
	}
</script>

<div class="metric-graph">
	{#if metricValues === null}
		<div class="empty">No metric selected</div>
	{:else if chartData.length === 0}
		<div class="empty">No metric data</div>
	{:else}
		<!-- The chart's hover/click only mirror the keyboard-accessible token grid. -->
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div
			class="chart"
			onclick={selectDatum}
			onmousemove={updateHoverIndex}
			onmouseleave={() => (hoveredIndex = null)}
		>
			<LineChart
				bind:context
				data={chartData}
				x="index"
				y="value"
				{yDomain}
				{series}
				padding={{ top: 0, right: 8, bottom: 24, left: 35 }}
				annotations={maskedAnnotations}
				props={{
					grid: { yTicks: Y_AXIS_TICK_COUNT },
					yAxis: {
						rule: true,
						ticks: Y_AXIS_TICK_COUNT,
						tickLabelProps: { dx: -8 }
					}
				}}
			>
				{#snippet highlight()}
					<Highlight
						data={selectedDatum}
						axis="x"
						points={false}
						motion="none"
						lines={{ class: 'metric-selected-focus-line' }}
					/>
					<Highlight
						data={hoveredDatum}
						axis="x"
						points={false}
						motion="none"
						lines={{ class: 'metric-hovered-focus-line' }}
					/>
				{/snippet}
				{#snippet tooltip()}{/snippet}
			</LineChart>
		</div>
	{/if}
</div>

<style>
	.metric-graph {
		height: 100%;
		width: 100%;
	}

	.chart {
		height: 100%;
		width: 100%;
		font-size: 0.7rem;
	}

	.empty {
		display: flex;
		height: 100%;
		align-items: center;
		justify-content: center;
		font-size: 0.75rem;
		color: var(--pico-muted-color);
	}

	/* layerchart renders SVG marks with `.lc-*` classes; without the Tailwind
	   chart-container we theme them here with Pico variables. */
	.metric-graph :global(.lc-spline-path),
	.metric-graph :global(.lc-line-path) {
		stroke: var(--pico-primary);
		stroke-width: 1.5px;
	}

	.metric-graph :global(.lc-axis-tick-label) {
		fill: var(--pico-muted-color);
		stroke: none;
		stroke-width: 0;
	}

	.metric-graph :global(.lc-axis-tick) {
		stroke: none;
	}

	.metric-graph :global(.lc-text-svg) {
		overflow: visible;
	}

	.metric-graph :global(.lc-grid-x-line),
	.metric-graph :global(.lc-grid-y-line),
	.metric-graph :global(.lc-rule-x-line),
	.metric-graph :global(.lc-rule-y-line) {
		stroke: var(--pico-border-color);
	}

	.metric-graph :global(.lc-axis.placement-left .lc-axis-rule) {
		stroke: var(--pico-muted-color);
		stroke-width: 1.5px;
	}

	.metric-graph :global(.lc-axis.placement-left .lc-axis-tick) {
		stroke: var(--pico-muted-color);
		stroke-width: 1px;
	}

	.metric-graph :global(.lc-highlight-line) {
		stroke-width: 0;
	}

	.metric-graph :global(.metric-masked-range) {
		fill: var(--pico-muted-border-color, var(--pico-border-color));
		opacity: 0.55;
		pointer-events: none;
	}

	.metric-graph :global(.metric-selected-focus-line) {
		stroke: var(--pico-color);
		stroke-width: 1.75px;
		stroke-dasharray: none;
	}

	.metric-graph :global(.metric-hovered-focus-line) {
		stroke: var(--pico-color);
		stroke-width: 1.5px;
		stroke-dasharray: 4 4;
	}
</style>
