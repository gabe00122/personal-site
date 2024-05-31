<script>
	export let url = '';
	export let description = '';
	export let altText = '';
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
