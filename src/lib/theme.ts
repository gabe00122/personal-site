import { writable } from 'svelte/store';
import { browser } from '$app/environment';

type Theme = 'light' | 'dark';

const userTheme = browser && localStorage.getItem('color-scheme');

export const theme = writable(userTheme ?? 'dark');

export const darkColor = 'rgb(194, 199, 208)';
export const lightColor = 'rgb(55, 60, 68)';

// update the theme
export function toggleTheme() {
	theme.update((currentTheme) => {
		const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

		document.documentElement.setAttribute('data-theme', newTheme);
		localStorage.setItem('color-scheme', newTheme);

		return newTheme;
	});
}
