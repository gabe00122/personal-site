<script lang="ts">
	import { tick } from 'svelte';
	import { darkColor, lightColor, theme } from '$lib/theme';
	import type { Episode } from './types';
	import { getPolicyTokenMask, isMaskedPolicyToken } from './policyMask';

	interface Props {
		episode: Episode;
		selectedIndex: number | null;
		hoveredIndex: number | null;
		metricKey: string;
		tokensHeight?: string;
	}

	type Theme = 'light' | 'dark';
	type MetricValues = ArrayLike<number>;

	const RAMP = {
		light: {
			seq: { l0: 0.96, l1: 0.8, c0: 0.02, c1: 0.13 },
			div: { l0: 0.96, l1: 0.82, c1: 0.13 }
		},
		dark: {
			seq: { l0: 0.22, l1: 0.46, c0: 0.01, c1: 0.12 },
			div: { l0: 0.21, l1: 0.46, c1: 0.12 }
		}
	} as const;

	const HUE_SEQ = 255;
	const HUE_NEG = 255;
	const HUE_POS = 40;

	let {
		episode,
		selectedIndex = $bindable(null),
		hoveredIndex = $bindable(null),
		metricKey: metricKey,
		tokensHeight = '18rem'
	}: Props = $props();

	let tokenElements = $state<HTMLElement[]>([]);
	let scrollContainer = $state<HTMLDivElement | null>(null);
	let colorTheme = $derived<Theme>($theme === 'light' ? 'light' : 'dark');
	let tokenTextColor = $derived(colorTheme === 'dark' ? darkColor : lightColor);
	let metricValues = $derived(getMetricValues(episode, metricKey));
	let policyTokenMask = $derived(getPolicyTokenMask(episode, metricKey));
	let metricRange = $derived(getMetricRange(metricValues));

	const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
	const cl01 = (x: number) => Math.min(1, Math.max(0, x));

	function getMetricValues(episode: Episode, metricKey: string): MetricValues | null {
		if (metricKey === 'none') {
			return null;
		}

		return episode.tokenMetrics[metricKey] ?? null;
	}

	function getMetricRange(values: MetricValues | null) {
		if (values === null) {
			return null;
		}

		let min = Infinity;
		let max = -Infinity;

		for (let i = 0; i < values.length; i++) {
			if (tokenIsMasked(i)) {
				continue;
			}

			const value = values[i];

			if (!Number.isFinite(value)) {
				continue;
			}

			min = Math.min(min, value);
			max = Math.max(max, value);
		}

		return min === Infinity ? null : { min, max };
	}

	function sequentialColor(value: number, min: number, max: number, theme: Theme, hue = HUE_SEQ) {
		if (max === min) {
			return 'transparent';
		}

		const colorRamp = RAMP[theme].seq;
		const ratio = cl01((value - min) / (max - min));
		return `oklch(${lerp(colorRamp.l0, colorRamp.l1, ratio)} ${lerp(colorRamp.c0, colorRamp.c1, ratio)} ${hue})`;
	}

	function divergingColor(value: number, min: number, max: number, theme: Theme) {
		const maxAbs = Math.max(Math.abs(min), Math.abs(max));
		if (maxAbs === 0) {
			return 'transparent';
		}

		const colorRamp = RAMP[theme].div;
		const scaledValue = Math.max(-1, Math.min(1, value / maxAbs));
		const magnitude = Math.abs(scaledValue);
		return `oklch(${lerp(colorRamp.l0, colorRamp.l1, magnitude)} ${colorRamp.c1 * magnitude} ${scaledValue >= 0 ? HUE_POS : HUE_NEG})`;
	}

	function getTokenColor(index: number) {
		const value = metricValues?.[index];
		if (
			tokenIsMasked(index) ||
			value === undefined ||
			!Number.isFinite(value) ||
			metricRange === null
		) {
			return 'transparent';
		}

		if (metricRange.min < 0 && metricRange.max > 0) {
			return divergingColor(value, metricRange.min, metricRange.max, colorTheme);
		}

		return sequentialColor(value, metricRange.min, metricRange.max, colorTheme);
	}

	function tokenIsMasked(index: number) {
		return isMaskedPolicyToken(policyTokenMask, index);
	}

	function selectToken(index: number) {
		if (tokenIsMasked(index)) {
			return;
		}

		selectedIndex = selectedIndex === index ? null : index;
	}

	function hoverToken(index: number | null) {
		if (index !== null && tokenIsMasked(index)) {
			return;
		}

		hoveredIndex = index;
	}

	function getDisplayToken(token: string) {
		return token.replace(/\r\n|\r|\n/g, '↵\n');
	}

	function handleKeydown(event: KeyboardEvent, index: number) {
		if (tokenIsMasked(index)) {
			return;
		}

		if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			selectToken(index);
		}
	}

	function scrollTokenIntoTokenPane(tokenElement: HTMLElement) {
		if (!scrollContainer) {
			return;
		}

		const tokenRect = tokenElement.getBoundingClientRect();
		const parentRect = scrollContainer.getBoundingClientRect();

		if (tokenRect.top < parentRect.top) {
			scrollContainer.scrollTop -= parentRect.top - tokenRect.top;
		} else if (tokenRect.bottom > parentRect.bottom) {
			scrollContainer.scrollTop += tokenRect.bottom - parentRect.bottom;
		}
	}

	$effect(() => {
		const index = selectedIndex;
		const tokenCount = episode.tokens.length;

		if (index === null || index < 0 || index >= tokenCount) {
			return;
		}

		tick().then(() => {
			const tokenElement = tokenElements[index];

			if (tokenElement) {
				scrollTokenIntoTokenPane(tokenElement);
			}
		});
	});
</script>

<div
	bind:this={scrollContainer}
	class="tokens-pane"
	style="--tokens-height: {tokensHeight}; --viz-token-text-color: {tokenTextColor};"
>
	<div class="tokens">
		{#each episode.tokens as token, index}
			{@const masked = tokenIsMasked(index)}
			<span
				bind:this={tokenElements[index]}
				tabindex={masked ? -1 : 0}
				role="button"
				aria-disabled={masked}
				aria-pressed={masked ? undefined : selectedIndex === index}
				class:hovered={!masked && hoveredIndex === index && selectedIndex !== index}
				class:selected={!masked && selectedIndex === index}
				class:masked
				style="--viz-token-color: {getTokenColor(index)};"
				onclick={() => selectToken(index)}
				onpointerenter={() => hoverToken(index)}
				onpointerleave={() => hoverToken(null)}
				onkeydown={(event) => handleKeydown(event, index)}>{getDisplayToken(token)}</span
			>
		{/each}
	</div>
</div>

<style>
	.tokens-pane {
		max-height: var(--tokens-height);
		overflow-y: scroll;
		padding: 0.75rem;
	}

	.tokens {
		line-height: 1.61;
	}

	.tokens span {
		font-family: var(--pico-font-family-monospace);

		/* Inline backgrounds only paint the glyph's content box, which is shorter
		   than the line box, so adjacent rows show a gap. Vertical padding grows
		   the background to fill the leading without shifting lines apart; `clone`
		   applies it to every fragment of a span that wraps across lines. */
		padding: 0.25em 0;
		box-decoration-break: clone;
		-webkit-box-decoration-break: clone;

		border: unset;
		border-radius: unset;
		outline: unset;
		box-shadow: unset;
		transition: unset;
		white-space: pre-wrap;

		background-color: var(--viz-token-color);
		color: var(--viz-token-text-color);
		cursor: pointer;
	}

	/* Hover/active: an accent ring that keeps the heatmap color visible. `inset`
	   so it never bleeds into neighbouring tokens and stays flush with the grid. */
	.tokens span:hover,
	.tokens span.hovered {
		box-shadow: inset 0 0 0 2px var(--pico-primary);
	}

	.tokens span.masked {
		cursor: default;
	}

	.tokens span.masked:hover,
	.tokens span.masked.hovered {
		box-shadow: unset;
	}

	/* Selected: the committed token drives the detail panel + graph marker, so
	   give it a solid accent fill with readable inverse text. Wins over hover via
	   source order (equal specificity). */
	.tokens span.selected {
		background-color: var(--pico-primary);
		color: var(--pico-primary-inverse);
	}

	.tokens span:focus-visible {
		outline: 2px solid var(--pico-primary);
		outline-offset: 2px;
	}
</style>
