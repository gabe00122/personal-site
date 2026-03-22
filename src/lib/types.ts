export type Categories = 'reinforcement learning' | 'jax' | 'transformer';

export type Post = {
	title: string;
	slug: string;
	description: string;
	date: string;
	categories: Categories[];
	published: boolean;
};
