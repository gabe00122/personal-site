<script lang="ts">
	import { formatDate } from '$lib/utils';
	import Seo from '$lib/components/seo.svelte';

	let { data } = $props();

	type TocItem = { id: string; text: string; level: 2 | 3 };

	let proseEl: HTMLElement | undefined = $state();
	let tocItems: TocItem[] = $state([]);
	let activeId = $state('');

	function headingLabel(heading: Element): string {
		const clone = heading.cloneNode(true) as HTMLElement;
		// The autolink anchor is empty, and KaTeX duplicates its text across the
		// MathML annotation and the aria-hidden HTML render — drop all three.
		clone.querySelectorAll('a, .katex-html, annotation').forEach((el) => el.remove());
		return clone.textContent?.replace(/\s+/g, ' ').trim() ?? '';
	}

	$effect(() => {
		data.content;
		if (!proseEl) return;
		const items = Array.from(proseEl.querySelectorAll<HTMLElement>('h2[id], h3[id]')).map((h) => ({
			id: h.id,
			text: headingLabel(h),
			level: h.tagName === 'H2' ? (2 as const) : (3 as const)
		}));
		tocItems = items;
		activeId = computeActive(items);
	});

	function computeActive(items: TocItem[]) {
		let current = '';
		for (const item of items) {
			const el = document.getElementById(item.id);
			if (el && el.getBoundingClientRect().top <= 120) current = item.id;
		}
		return current;
	}

	function onScroll() {
		activeId = computeActive(tocItems);
	}
</script>

<svelte:window onscroll={onScroll} />

<Seo title={data.meta.title} description={data.meta.description} type="article" />

<div class="content-column">
	<article class="post">
		<hgroup class="post-header">
			<h1 class="post-title">{data.meta.title}</h1>
			<p><time class="meta" datetime={data.meta.date}>{formatDate(data.meta.date)}</time></p>
		</hgroup>

		<div class="prose" bind:this={proseEl}>
			<data.content />
		</div>
	</article>
</div>

{#if tocItems.length >= 2}
	<nav class="toc" aria-label="Table of contents">
		<div class="toc-label meta">Contents</div>
		<ul>
			{#each tocItems as item (item.id)}
				<li class:sub={item.level === 3} class:active={activeId === item.id}>
					<a href={'#' + item.id}>{item.text}</a>
				</li>
			{/each}
		</ul>
	</nav>
{/if}

<style>
	.post {
		margin: 0;
		padding: 0;
		background: none;
		box-shadow: none;
	}
	.post-header {
		margin-bottom: 2rem;
	}
	.post-title {
		font-size: 2rem;
		line-height: 1.25;
		margin-bottom: 0.5rem;
	}

	.toc {
		display: none;
	}

	@media (min-width: 1280px) {
		.toc {
			display: block;
			position: fixed;
			top: 7rem;
			left: calc(50% + 22.5rem);
			width: min(16rem, calc(50vw - 22.5rem - 1.5rem));
			max-height: calc(100vh - 10rem);
			overflow-y: auto;
			overflow-x: hidden;
			scrollbar-width: thin;
		}
	}

	.toc-label {
		text-transform: uppercase;
		letter-spacing: 0.08em;
		margin-bottom: 0.75rem;
	}

	/* Pico lays out nav ul/li as a flex row; this is a vertical list */
	.toc ul {
		display: block;
		list-style: none;
		margin: 0;
		padding: 0;
		border-left: 1px solid var(--pico-muted-border-color);
	}

	.toc li {
		display: block;
		margin: 0;
		padding: 0.2rem 0 0.2rem 0.9rem;
		border-left: 2px solid transparent;
		margin-left: -1.5px;
	}

	.toc li.sub {
		padding-left: 1.9rem;
	}

	.toc li.active {
		border-left-color: var(--pico-primary);
	}

	.toc a {
		display: inline-block;
		padding: 0;
		font-size: 0.875rem;
		line-height: 1.45;
		color: var(--pico-muted-color);
		text-decoration: none;
	}

	.toc a:hover,
	.toc a:focus-visible {
		color: var(--pico-color);
	}

	.toc li.active > a {
		color: var(--pico-primary);
	}
</style>
