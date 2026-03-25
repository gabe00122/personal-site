<script lang="ts">
	import Play from 'lucide-svelte/icons/play';
	import Pause from 'lucide-svelte/icons/pause';
	import Maximize from 'lucide-svelte/icons/maximize';
	import Minimize from 'lucide-svelte/icons/minimize';

	import { theme } from '$lib/theme';

	interface Props {
		url: string;
		description: string;
	}

	let { url, description }: Props = $props();

	let videoEl: HTMLVideoElement | undefined = $state();
	let wrapperEl: HTMLDivElement | undefined = $state();
	let paused = $state(true);
	let seeking = $state(false);
	let currentTime = $state(0);
	let duration = $state(0);
	let isFullscreen = $state(false);

	let buttonFillColor = $derived($theme === 'dark' ? 'rgb(194, 199, 208)' : 'rgb(55, 60, 68)');

	$effect(() => {
		if (!videoEl) return;
		if (paused || seeking) {
			videoEl.pause();
		} else {
			videoEl.play();
		}
	});

	function togglePlay() {
		paused = !paused;
	}

	function seek(e: Event) {
		const value = (e.target as HTMLInputElement).valueAsNumber;
		currentTime = value;
	}

	function toggleFullscreen() {
		if (!wrapperEl) return;
		if (document.fullscreenElement) {
			document.exitFullscreen();
		} else {
			wrapperEl.requestFullscreen();
		}
	}

	function onFullscreenChange() {
		isFullscreen = !!document.fullscreenElement;
	}

	function formatTime(seconds: number): string {
		const m = Math.floor(seconds / 60);
		const s = Math.floor(seconds % 60);
		return `${m}:${s.toString().padStart(2, '0')}`;
	}

	function seekStart() {
		seeking = true;
	}

	function seekEnd() {
		seeking = false;
	}
</script>

<figure class="centered">
	<figcaption id="video-caption" class="centered-text">{description}</figcaption>
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div class="video-wrapper" bind:this={wrapperEl} onfullscreenchange={onFullscreenChange}>
		<video
			bind:this={videoEl}
			bind:currentTime
			bind:duration
			muted
			aria-describedby="video-caption"
			onclick={togglePlay}
		>
			<source src={url} type="video/mp4" />
		</video>
		<div class="controls">
			<button class="btn" onclick={togglePlay} aria-label={paused ? 'Play' : 'Pause'}>
				{#if paused}
					<Play size={16} color={buttonFillColor} fill={buttonFillColor} />
				{:else}
					<Pause size={16} color={buttonFillColor} fill={buttonFillColor} />
				{/if}
			</button>
			<input
				type="range"
				class="progress"
				min="0"
				max={duration || 0}
				step="0.001"
				value={currentTime}
				oninput={seek}
				onpointerdown={seekStart}
				onpointerup={seekEnd}
				aria-label="Seek"
			/>
			<span class="time">{formatTime(currentTime)} / {formatTime(duration)}</span>
			<button
				class="btn"
				onclick={toggleFullscreen}
				aria-label={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
			>
				{#if isFullscreen}
					<Minimize size={16} color={buttonFillColor} fill={buttonFillColor} />
				{:else}
					<Maximize size={16} color={buttonFillColor} fill={buttonFillColor} />
				{/if}
			</button>
		</div>
	</div>
</figure>

<style>
	.centered {
		width: fit-content;
		max-width: 100%;
		margin-left: auto;
		margin-right: auto;
		margin-bottom: 1rem;
	}

	.video-wrapper {
		display: flex;
		flex-direction: column;
		overflow: hidden;
		border-radius: var(--pico-border-radius);
		border: 1px solid var(--pico-muted-border-color);
	}

	.video-wrapper:fullscreen {
		background-color: var(--pico-background-color);
	}

	.video-wrapper:fullscreen video {
		flex: 1;
		min-height: 0;
		width: 100%;
		height: 100%;
		object-fit: contain;
	}

	.centered-text {
		text-align: center;
	}

	.controls {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.4rem 0.5rem;
		border-top: 1px solid var(--pico-muted-border-color);
	}

	.btn {
		all: unset;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 0.15rem;
	}

	.btn:hover {
		opacity: 0.7;
	}

	.progress {
		flex: 1;
		cursor: pointer;
		margin: 0;
	}

	.time {
		font-size: 0.75rem;
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
		color: var(--pico-color);
	}
</style>
