<script lang="ts">
	interface Props {
		url: string;
		description: string;
	}

	let { url, description }: Props = $props();

	const uid = $props.id();
	let poster = $derived(url.replace(/\.mp4$/, '-poster.webp'));
</script>

<figure class="video-container">
	<figcaption id="video-caption-{uid}" class="centered-text">{description}</figcaption>
	<video playsinline controls muted aria-describedby="video-caption-{uid}" {poster}>
		<source src={url} type="video/mp4" />
	</video>
</figure>

<style>
	.video-container {
		margin-bottom: 1rem;
	}

	.video-container video {
		max-width: 100%;
		border-radius: var(--pico-border-radius);
		display: block;
		margin-left: auto;
		margin-right: auto;
	}

	.centered-text {
		text-align: center;
	}
</style>
