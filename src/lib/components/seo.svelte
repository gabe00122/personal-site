<script lang="ts">
	import { page } from '$app/state';
	import { title as siteTitle } from '$lib/config';

	interface Props {
		title?: string;
		description?: string;
		type?: 'website' | 'article';
	}

	let { title, description, type = 'website' }: Props = $props();

	let fullTitle = $derived(title ? `${title} | ${siteTitle}` : siteTitle);
	let imageUrl = $derived(new URL('/og-image.png', page.url.origin).href);
</script>

<svelte:head>
	<title>{fullTitle}</title>
	{#if description}
		<meta name="description" content={description} />
		<meta property="og:description" content={description} />
	{/if}
	<meta property="og:type" content={type} />
	<meta property="og:title" content={title ?? siteTitle} />
	<meta property="og:site_name" content={siteTitle} />
	<meta property="og:url" content={page.url.href} />
	<meta property="og:image" content={imageUrl} />
	<meta name="twitter:card" content="summary" />
</svelte:head>
