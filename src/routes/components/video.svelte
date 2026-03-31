<script lang="ts">
	interface Props {
		url: string;
		description: string;
	}

	let { url, description }: Props = $props();
</script>

<figure class="video-container">
	<figcaption id="video-caption" class="centered-text">{description}</figcaption>
	<video playsinline controls preload="metadata" muted aria-describedby="video-caption">
		<source src={url} type="video/mp4" />
	</video>
</figure>

<style>
	.video-container {
		margin-bottom: 1rem;
	}

	.video-container video {
		max-width: 100%;
	}

	.centered-text {
		text-align: center;
	}
</style>
