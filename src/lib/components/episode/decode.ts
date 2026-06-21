import type { EncodedEpisode, Episode } from './types';

export function base64Decode(data: string): Float32Array {
	const binaryString = atob(data);
	const bytes = new Uint8Array(binaryString.length);

	for (let i = 0; i < binaryString.length; i++) {
		bytes[i] = binaryString.charCodeAt(i);
	}

	return new Float32Array(bytes.buffer);
}

export function decodeEpisode(episode: EncodedEpisode): Episode {
	const { tokens, tokenMetrics } = episode;

	const decodedMetrics = Object.fromEntries(
		Object.entries(tokenMetrics).map(([name, value]) => [name, base64Decode(value)])
	);

	return {
		tokens,
		tokenMetrics: decodedMetrics
	};
}
