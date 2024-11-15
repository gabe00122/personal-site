<script lang="ts">
	interface Props {
		url?: string;
		description?: string;
		altText?: string;
	}

	let { url = '', description = '', altText = '' }: Props = $props();
</script>

<figure>
	<figcaption class="centered-text">{description}</figcaption>
	<img src={url} alt={altText} />
</figure>

<style>
	.centered-text {
		text-align: center;
	}
</style>
