import adapter from '@sveltejs/adapter-auto';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

import { mdsvex, escapeSvelte } from 'mdsvex';
import { getHighlighter } from 'shiki';

import remarkUnwrapImages from 'remark-unwrap-images'
import rehypeSlug from 'rehype-slug';

import rehypeKatexSvelte from "rehype-katex-svelte";
import remarkMath from 'remark-math'

const highlighterTheme = 'dracula';
const highlighterLanguages = ['javascript', 'typescript', 'python'];

const mdsvexOptions = {
	extensions: ['.md'],
	highlight: {
		highlighter: async (code, lang = 'text') => {
			const highlighter = await getHighlighter({
				themes: [highlighterTheme],
				langs: highlighterLanguages,
			})
			await highlighter.loadLanguage(...highlighterLanguages)
			const html = escapeSvelte(highlighter.codeToHtml(code, { lang, theme: highlighterTheme }))
			return `{@html \`${html}\` }`
		}
	},
	remarkPlugins: [remarkMath, remarkUnwrapImages],
	rehypePlugins: [rehypeKatexSvelte, rehypeSlug]
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
