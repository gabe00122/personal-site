export function humanizeMetricKey(metricKey: string) {
	return metricKey
		.replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
		.replace(/([a-z0-9])([A-Z])/g, '$1 $2')
		.replace(/[-_]+/g, ' ')
		.trim()
		.replace(/\s+/g, ' ')
		.replace(/\b\w/g, (character) => character.toUpperCase());
}

export function metricOptionLabel(metricKey: string) {
	return metricKey === 'none' ? 'None' : humanizeMetricKey(metricKey);
}

export function metricDetailLabel(metricKey: string) {
	return metricKey === 'none' ? 'Metric' : humanizeMetricKey(metricKey);
}
