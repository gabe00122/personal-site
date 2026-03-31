<script lang="ts">
	import Play from 'lucide-svelte/icons/play';
	import Pause from 'lucide-svelte/icons/pause';
	import Maximize from 'lucide-svelte/icons/maximize';
	import Minimize from 'lucide-svelte/icons/minimize';

	import { theme, darkColor, lightColor } from '$lib/theme';
	import { isMobile } from '$lib/utils';

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

	let buttonFillColor = $derived($theme === 'dark' ? darkColor : lightColor);

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

<figure class="video-container">
	<figcaption id="video-caption" class="centered-text">{description}</figcaption>
	<div class="video-wrapper" bind:this={wrapperEl} onfullscreenchange={onFullscreenChange}>
		<video
			bind:this={videoEl}
			bind:currentTime
			bind:duration
			playsinline
			muted
			aria-describedby="video-caption"
			onclick={togglePlay}
		>
			<source src={url} type="video/mp4" />
		</video>
		{#if !isMobile()}
			<div class="controls">
				<button
					class="btn outline contrast"
					onclick={togglePlay}
					aria-label={paused ? 'Play' : 'Pause'}
				>
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
					class="btn contrast outline"
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
		{/if}
	</div>
</figure>

<style>
	.video-container {
		margin-bottom: 1rem;
	}

	.video-wrapper {
		margin-left: auto;
		margin-right: auto;
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
	}

	.centered-text {
		text-align: center;
	}

	.controls {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.4rem 0.5rem;
	}

	.btn {
		padding: 0.5rem;
		display: flex;
		border: none;
	}

	.btn:hover {
		opacity: 0.7;
	}

	.progress {
		cursor: pointer;
		margin: 0;
	}

	.time {
		font-size: 0.75rem;
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
	}
</style>
