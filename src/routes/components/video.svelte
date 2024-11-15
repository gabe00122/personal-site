<script lang="ts">
	interface Props {
		url: string;
		description: string;
	}

	let { url, description }: Props = $props();
</script>

<figure class="centered">
	<figcaption id="video-caption" class="centered-text">{description}</figcaption>
	<video autoplay loop muted playsinline aria-describedby="video-caption" class="bordered">
		<source src={url} type="video/mp4" />
	</video>
</figure>

<style>
	.centered {
		width: fit-content;
		margin-left: auto;
		margin-right: auto;
	}

	.centered video {
		max-width: 100%;
	}

	.bordered {
		border: 1px solid black;
	}

	.centered-text {
		text-align: center;
	}
</style>
