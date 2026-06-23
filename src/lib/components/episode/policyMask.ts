import type { Episode } from './types';

const POLICY_TOKEN_METRICS = new Set(['actor_loss', 'advantage', 'log_probs']);

export function usesPolicyTokenMask(metricKey: string) {
	return POLICY_TOKEN_METRICS.has(metricKey);
}

export function getPolicyTokenMask(episode: Episode, metricKey: string): ArrayLike<number> | null {
	if (!usesPolicyTokenMask(metricKey)) {
		return null;
	}

	return episode.tokenMetrics.policy_mask ?? null;
}

export function isMaskedPolicyToken(mask: ArrayLike<number> | null, index: number) {
	return mask !== null && !(mask[index] > 0);
}

export function firstUnmaskedPolicyToken(mask: ArrayLike<number> | null, length: number) {
	if (mask === null) {
		return length > 0 ? 0 : null;
	}

	for (let i = 0; i < length; i++) {
		if (!isMaskedPolicyToken(mask, i)) {
			return i;
		}
	}

	return null;
}
