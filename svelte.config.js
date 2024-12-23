import adapter from '@sveltejs/adapter-auto';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

import { mdsvex, escapeSvelte } from 'mdsvex';
import { getSingletonHighlighter as getHighlighter } from 'shiki';

import remarkUnwrapImages from 'remark-unwrap-images'
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';

import rehypeKatexSvelte from "rehype-katex-svelte";
import remarkMath from 'remark-math'

const highlighterTheme = 'dracula';
const highlighterLanguages = ['javascript', 'typescript', 'python'];
const highlighter = await getHighlighter({
	themes: [highlighterTheme],
	langs: highlighterLanguages,
})

const mdsvexOptions = {
	extensions: ['.md'],
	highlight: {
		highlighter: async (code, lang = 'text') => {
			await highlighter.loadLanguage(...highlighterLanguages)
			const html = escapeSvelte(highlighter.codeToHtml(code, { lang, theme: highlighterTheme }))
			return `{@html \`${html}\` }`
		}
	},
	remarkPlugins: [remarkMath, remarkUnwrapImages],
	rehypePlugins: [rehypeKatexSvelte, rehypeSlug, rehypeAutolinkHeadings]
}

/** @type {import('@sveltejs/kit').Config} */
const config = {
	extensions: ['.svelte', '.md'],
	preprocess: [vitePreprocess(), mdsvex(mdsvexOptions)],
	kit: {
		adapter: adapter()
	},
};

export default config;
