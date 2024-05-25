<script lang="ts">
	export let url: string;
	export let description: string;
</script>

<figure class="centered">
	<figcaption class="centered-text">{description}</figcaption>
	<video autoplay loop muted aria-describedby="" class="bordered">
		<source src={url} type="video/mp4" />
	</video>
</figure>

<style>
	.centered {
		display: block;
		max-width: fit-content;
		height: auto;
		margin-left: auto;
		margin-right: auto;
	}

	.bordered {
		border: 1px solid black;
	}

	.centered-text {
		text-align: center;
	}
</style>
