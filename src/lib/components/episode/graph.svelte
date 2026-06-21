<script lang="ts">
	import type { Episode } from './types';
	import { Highlight, LineChart, type ChartState } from 'layerchart';

	interface Props {
		episode: Episode;
		metricKey: string;
		selectedIndex: number | null;
		hoveredIndex: number | null;
	}

	type MetricDatum = {
		index: number;
		value: number;
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
	let chartData = $derived(getChartData(metricValues));
	let yDomain = $derived(getYDomain(chartData));
	let hoveredDatum = $derived(getDatum(hoveredIndex));
	let selectedDatum = $derived(getDatum(selectedIndex));
	let series = $derived([
		{
			key: 'value',
			label: metricKey === 'none' ? 'Metric' : metricKey,
			color: 'var(--pico-primary)'
		}
	]);

	function getChartData(values: ArrayLike<number> | null): MetricDatum[] {
		if (values === null) {
			return [];
		}

		const out: MetricDatum[] = [];
		for (let i = 0; i < values.length; i++) {
			out.push({ index: i, value: values[i] });
		}
		return out;
	}

	function getDatum(index: number | null) {
		if (index === null) {
			return undefined;
		}
		return chartData[index];
	}

	function getYDomain(data: MetricDatum[]): [number, number] | undefined {
		if (data.length === 0) {
			return undefined;
		}

		let min = 10;
		let max = -10;
		for (const { value } of data) {
			min = Math.min(min, value);
			max = Math.max(max, value);
		}

		const span = max - min;
		const padding = span === 0 ? 1 : span * 0.08;
		return [min - padding, max + padding];
	}

	function tooltipIndex() {
		return (context?.tooltipState.data as Partial<MetricDatum> | null)?.index ?? null;
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
		<div class="chart" onclick={selectDatum} onmousemove={updateHoverIndex}>
			<LineChart
				bind:context
				data={chartData}
				x="index"
				y="value"
				{yDomain}
				{series}
				padding={{ top: 0, right: 8, bottom: 24, left: 30 }}
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
