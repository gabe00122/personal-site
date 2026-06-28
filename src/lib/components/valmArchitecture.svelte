<script lang="ts">
	interface Props {
		/** Optional caption shown beneath the diagram. */
		caption?: string;
	}

	let {
		caption = 'The token embedding feeds the policy, while the token embedding and the last reward ' +
			'both feed the value stream — the reward never enters the policy. Every n-th base latent is ' +
			'also encoded into the value stream, where the encode taps are stop-gradient, so the policy ' +
			'and value gradients never cross.'
	}: Props = $props();
</script>

<figure class="arch-figure">
	<svg
		class="arch"
		viewBox="0 0 560 730"
		role="img"
		aria-label="A ladder value network running parallel to the Qwen3 base model. Grey arrows show
			the forward pass flowing up: the token embedding flows up through three base layers to the
			token prediction, while the token embedding and the last reward together feed a value stream
			that flows up through three value layers to the value prediction, each value layer also
			receiving an encoded latent tapped from a base layer. Green arrows show the policy gradient
			flowing back down through the base layers; red arrows show the value gradient flowing back
			down through the value layers. The two backward streams never cross."
		xmlns="http://www.w3.org/2000/svg"
	>
		<defs>
			<marker
				id="arch-arrow"
				viewBox="0 0 10 10"
				refX="8.5"
				refY="5"
				markerWidth="7.5"
				markerHeight="7.5"
				orient="auto"
			>
				<path d="M0,0 L10,5 L0,10 z" fill="context-stroke" />
			</marker>
		</defs>

		<!-- ───── forward pass (grey, flows up) ───── -->
		<g class="fwd">
			<!-- base residual stream -->
			<line x1="180" y1="665" x2="180" y2="553" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="481" x2="180" y2="398" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="326" x2="180" y2="243" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="171" x2="180" y2="52" marker-end="url(#arch-arrow)" />

			<!-- value residual stream (bottom segment rises from the input merge) -->
			<line x1="420" y1="620" x2="420" y2="486" marker-end="url(#arch-arrow)" />
			<line x1="420" y1="434" x2="420" y2="331" marker-end="url(#arch-arrow)" />
			<line x1="420" y1="279" x2="420" y2="176" marker-end="url(#arch-arrow)" />
			<line x1="420" y1="124" x2="420" y2="52" marker-end="url(#arch-arrow)" />

			<!-- value-encode taps: base latent -> value layer -->
			<line x1="180" y1="150" x2="358" y2="150" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="305" x2="358" y2="305" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="460" x2="358" y2="460" marker-end="url(#arch-arrow)" />

			<!-- token embedding feeds the value net (and, separately, the policy above) -->
			<line x1="180" y1="620" x2="420" y2="620" marker-end="url(#arch-arrow)" />
			<!-- last reward feeds only the value net -->
			<path d="M470,665 L470,600 L420,600" fill="none" marker-end="url(#arch-arrow)" />
		</g>

		<!-- ───── policy backward pass (green, flows down the base stream) ───── -->
		<g class="bwd-policy">
			<line x1="160" y1="52" x2="160" y2="171" marker-end="url(#arch-arrow)" />
			<line x1="160" y1="243" x2="160" y2="326" marker-end="url(#arch-arrow)" />
			<line x1="160" y1="398" x2="160" y2="481" marker-end="url(#arch-arrow)" />
		</g>

		<!-- ───── value backward pass (red, flows down the value stream) ───── -->
		<g class="bwd-value">
			<line x1="400" y1="52" x2="400" y2="124" marker-end="url(#arch-arrow)" />
			<line x1="400" y1="176" x2="400" y2="279" marker-end="url(#arch-arrow)" />
			<line x1="400" y1="331" x2="400" y2="434" marker-end="url(#arch-arrow)" />
		</g>

		<!-- forward junction dots -->
		<g class="junction">
			<circle cx="180" cy="150" r="3.5" />
			<circle cx="180" cy="305" r="3.5" />
			<circle cx="180" cy="460" r="3.5" />
			<circle cx="180" cy="620" r="3.5" />
			<circle cx="420" cy="620" r="3.5" />
			<circle cx="420" cy="600" r="3.5" />
		</g>

		<!-- ───── base layers (green boxes) ───── -->
		<g class="box base">
			<rect x="95" y="175" width="150" height="64" rx="12" />
			<rect x="95" y="330" width="150" height="64" rx="12" />
			<rect x="95" y="485" width="150" height="64" rx="12" />
		</g>
		<g class="box-label">
			<text x="170" y="211">Base layer</text>
			<text x="170" y="366">Base layer</text>
			<text x="170" y="521">Base layer</text>
		</g>

		<!-- ───── value layers (red boxes) ───── -->
		<g class="box value">
			<rect x="362" y="128" width="96" height="44" rx="9" />
			<rect x="362" y="283" width="96" height="44" rx="9" />
			<rect x="362" y="438" width="96" height="44" rx="9" />
		</g>
		<g class="box-label small">
			<text x="410" y="154">Value layer</text>
			<text x="410" y="309">Value layer</text>
			<text x="410" y="464">Value layer</text>
		</g>

		<!-- ───── endpoint labels ───── -->
		<text class="label base-label" x="170" y="34">Token Prediction</text>
		<text class="label value-label" x="410" y="34">Value</text>
		<text class="label" x="165" y="700">Token Embedding</text>
		<text class="label" x="470" y="700">Last Reward</text>
	</svg>

	<ul class="legend">
		<li class="fwd">
			<svg class="swatch" viewBox="0 0 30 12" aria-hidden="true">
				<line x1="2" y1="6" x2="20" y2="6" />
				<path d="M19,2 L28,6 L19,10 Z" />
			</svg>
			<span>Forward pass</span>
		</li>
		<li class="policy">
			<svg class="swatch" viewBox="0 0 30 12" aria-hidden="true">
				<line x1="2" y1="6" x2="20" y2="6" />
				<path d="M19,2 L28,6 L19,10 Z" />
			</svg>
			<span>Policy gradient</span>
		</li>
		<li class="value">
			<svg class="swatch" viewBox="0 0 30 12" aria-hidden="true">
				<line x1="2" y1="6" x2="20" y2="6" />
				<path d="M19,2 L28,6 L19,10 Z" />
			</svg>
			<span>Value gradient</span>
		</li>
	</ul>

	{#if caption}
		<figcaption>{caption}</figcaption>
	{/if}
</figure>

<style>
	.arch-figure {
		margin-block: var(--pico-block-spacing-vertical, 1.5rem);
		margin-inline: 0;

		/* Forward pass + the two backward gradient streams. Shared by the
		   diagram and the legend below it. */
		--fwd: #9aa1ab;
		--base: #4faa3c;
		--value: #d2473c;
		--label: var(--pico-color);
	}

	/* Deeper accents read better on the light background. */
	:global([data-theme='light']) .arch-figure {
		--fwd: #5b626d;
		--base: #3c8f2c;
		--value: #c23a2f;
	}

	.arch {
		display: block;
		width: 100%;
		max-width: 30rem;
		height: auto;
		margin-inline: auto;
	}

	.fwd line,
	.fwd path,
	.bwd-policy line,
	.bwd-value line {
		stroke-width: 2;
		fill: none;
	}

	.fwd line,
	.fwd path {
		stroke: var(--fwd);
	}

	.bwd-policy line {
		stroke: var(--base);
	}

	.bwd-value line {
		stroke: var(--value);
	}

	.junction circle {
		fill: var(--fwd);
	}

	.box rect {
		stroke-width: 1.5;
	}

	.box.base rect {
		fill: var(--base);
		stroke: color-mix(in srgb, var(--base) 65%, #000);
	}

	.box.value rect {
		fill: var(--value);
		stroke: color-mix(in srgb, var(--value) 65%, #000);
	}

	.box-label text {
		fill: #fff;
		font-size: 15px;
		font-weight: 600;
		text-anchor: middle;
		dominant-baseline: middle;
	}

	.box-label.small text {
		font-size: 13px;
	}

	.label {
		fill: var(--label);
		font-size: 15px;
		text-anchor: middle;
	}

	.label.base-label {
		fill: var(--base);
	}

	.label.value-label {
		fill: var(--value);
	}

	.legend {
		display: flex;
		flex-wrap: wrap;
		justify-content: center;
		gap: 0.5rem 1.25rem;
		max-width: 30rem;
		margin: 0.75rem auto 0;
		padding: 0;
		list-style: none;
		font-size: 0.85rem;
		color: var(--pico-muted-color);
	}

	.legend li {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
	}

	.swatch {
		width: 30px;
		height: 12px;
		flex: none;
	}

	.swatch line {
		stroke: currentColor;
		stroke-width: 2;
	}

	.swatch path {
		fill: currentColor;
	}

	.legend li.fwd .swatch {
		color: var(--fwd);
	}

	.legend li.policy .swatch {
		color: var(--base);
	}

	.legend li.value .swatch {
		color: var(--value);
	}

	figcaption {
		max-width: 30rem;
		margin: 0.5rem auto 0;
		text-align: center;
		font-size: 0.85rem;
		color: var(--pico-muted-color);
	}
</style>
