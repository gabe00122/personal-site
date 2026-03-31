<script lang="ts">
	interface Props {
		url?: string;
		description?: string;
		altText?: string;
		align?: 'left' | 'center' | 'right';
	}

	let { url = '', description = '', altText = '', align }: Props = $props();
</script>

<figure>
	<figcaption class="centered-text">{description}</figcaption>
	<img
		class="img"
		src={url}
		alt={altText}
		class:centered={align === 'center'}
		class:right={align === 'right'}
	/>
</figure>

<style>
	.centered-text {
		text-align: center;
	}

	.img {
		border-radius: var(--pico-border-radius);
	}

	img.centered {
		display: block;
		margin-left: auto;
		margin-right: auto;
	}

	img.right {
		display: block;
		margin-left: auto;
		margin-right: 0;
	}
</style>
