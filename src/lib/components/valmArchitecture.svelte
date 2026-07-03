<script lang="ts">
	interface Props {
		/** Optional caption shown beneath the diagram. */
		caption?: string;
	}

	let {
		caption = 'The base layers (frozen Qwen3 blocks with LoRA adapters) form the policy, running ' +
			'from the token embedding up to the token prediction. A scaled-down transformer with fewer ' +
			'layers runs alongside as the value network, fed by the token embedding and the last reward ' +
			'— the reward never enters the policy. The token embedding and every n-th base latent are ' +
			'projected into the value stream through SwiGLU value-encode blocks; the ⫽ marks are ' +
			'stop-gradients, so the policy and value gradients never cross.'
	}: Props = $props();
</script>

<figure class="arch-figure">
	<svg
		class="arch"
		viewBox="0 0 560 810"
		role="img"
		aria-label="A ladder value network running parallel to the Qwen3 base model. Grey arrows show
			the forward pass flowing up: the token embedding flows up through a stack of six base layers,
			each a frozen Qwen3 block with a LoRA adapter, to the token prediction. The token embedding, tapped through
			its own stop-gradient value-encode block, and the last reward together feed a parallel
			value stream of three smaller value layers
			ending in the value prediction. The output of every second base layer is tapped through a
			value-encode block into a value layer; a stop-gradient mark on each tap shows that value
			gradients cannot flow back into the base model. Green arrows show the policy gradient
			flowing back down the base stack; red arrows show the value gradient flowing back down the
			value stack. The two backward streams never cross."
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
			<line x1="180" y1="758" x2="180" y2="678" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="620" x2="180" y2="582" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="524" x2="180" y2="486" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="428" x2="180" y2="390" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="332" x2="180" y2="294" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="236" x2="180" y2="198" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="140" x2="180" y2="54" marker-end="url(#arch-arrow)" />

			<!-- value residual stream (bottom segment rises from the input merge) -->
			<line x1="430" y1="726" x2="430" y2="626" marker-end="url(#arch-arrow)" />
			<line x1="430" y1="576" x2="430" y2="434" marker-end="url(#arch-arrow)" />
			<line x1="430" y1="384" x2="430" y2="242" marker-end="url(#arch-arrow)" />
			<line x1="430" y1="192" x2="430" y2="54" marker-end="url(#arch-arrow)" />

			<!-- taps: every 2nd base latent -> value encode -> value layer -->
			<line x1="180" y1="215" x2="376" y2="215" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="407" x2="376" y2="407" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="599" x2="376" y2="599" marker-end="url(#arch-arrow)" />

			<!-- token embedding feeds the value net (and, separately, the policy above) -->
			<line x1="180" y1="726" x2="426" y2="726" marker-end="url(#arch-arrow)" />
			<!-- last reward feeds only the value net -->
			<path d="M490,758 L490,700 L434,700" fill="none" marker-end="url(#arch-arrow)" />
		</g>

		<!-- ───── policy backward pass (green, flows down the base stream) ───── -->
		<g class="bwd-policy">
			<line x1="160" y1="54" x2="160" y2="136" marker-end="url(#arch-arrow)" />
			<line x1="160" y1="194" x2="160" y2="232" marker-end="url(#arch-arrow)" />
			<line x1="160" y1="290" x2="160" y2="328" marker-end="url(#arch-arrow)" />
			<line x1="160" y1="386" x2="160" y2="424" marker-end="url(#arch-arrow)" />
			<line x1="160" y1="482" x2="160" y2="520" marker-end="url(#arch-arrow)" />
			<line x1="160" y1="578" x2="160" y2="616" marker-end="url(#arch-arrow)" />
		</g>

		<!-- ───── value backward pass (red, flows down the value stream) ───── -->
		<g class="bwd-value">
			<line x1="410" y1="54" x2="410" y2="188" marker-end="url(#arch-arrow)" />
			<line x1="410" y1="238" x2="410" y2="380" marker-end="url(#arch-arrow)" />
			<line x1="410" y1="430" x2="410" y2="572" marker-end="url(#arch-arrow)" />
		</g>

		<!-- stop-gradient marks on the taps -->
		<g class="sg">
			<line x1="214" y1="223" x2="222" y2="207" />
			<line x1="223" y1="223" x2="231" y2="207" />
			<line x1="214" y1="415" x2="222" y2="399" />
			<line x1="223" y1="415" x2="231" y2="399" />
			<line x1="214" y1="607" x2="222" y2="591" />
			<line x1="223" y1="607" x2="231" y2="591" />
			<line x1="214" y1="734" x2="222" y2="718" />
			<line x1="223" y1="734" x2="231" y2="718" />
		</g>

		<!-- forward junction dots -->
		<g class="junction">
			<circle cx="180" cy="215" r="3.5" />
			<circle cx="180" cy="407" r="3.5" />
			<circle cx="180" cy="599" r="3.5" />
			<circle cx="180" cy="726" r="3.5" />
			<circle cx="430" cy="726" r="3.5" />
			<circle cx="430" cy="700" r="3.5" />
		</g>

		<!-- ───── base layers (green boxes) ───── -->
		<g class="box base">
			<rect x="105" y="140" width="150" height="54" rx="12" />
			<rect x="105" y="236" width="150" height="54" rx="12" />
			<rect x="105" y="332" width="150" height="54" rx="12" />
			<rect x="105" y="428" width="150" height="54" rx="12" />
			<rect x="105" y="524" width="150" height="54" rx="12" />
			<rect x="105" y="620" width="150" height="54" rx="12" />
		</g>
		<g class="box-label">
			<text x="180" y="160">Base layer</text>
			<text x="180" y="256">Base layer</text>
			<text x="180" y="352">Base layer</text>
			<text x="180" y="448">Base layer</text>
			<text x="180" y="544">Base layer</text>
			<text x="180" y="640">Base layer</text>
		</g>
		<g class="box-sublabel">
			<text x="180" y="181">frozen + LoRA</text>
			<text x="180" y="277">frozen + LoRA</text>
			<text x="180" y="373">frozen + LoRA</text>
			<text x="180" y="469">frozen + LoRA</text>
			<text x="180" y="565">frozen + LoRA</text>
			<text x="180" y="661">frozen + LoRA</text>
		</g>

		<!-- ───── value encode blocks (on the taps) ───── -->
		<g class="box encode">
			<rect x="250" y="198" width="84" height="34" rx="8" />
			<rect x="250" y="390" width="84" height="34" rx="8" />
			<rect x="250" y="582" width="84" height="34" rx="8" />
			<rect x="250" y="709" width="84" height="34" rx="8" />
		</g>
		<g class="encode-label">
			<text x="292" y="215">Value encode</text>
			<text x="292" y="407">Value encode</text>
			<text x="292" y="599">Value encode</text>
			<text x="292" y="726">Value encode</text>
		</g>

		<!-- ───── value layers (red boxes) ───── -->
		<g class="box value">
			<rect x="380" y="192" width="100" height="46" rx="9" />
			<rect x="380" y="384" width="100" height="46" rx="9" />
			<rect x="380" y="576" width="100" height="46" rx="9" />
		</g>
		<g class="box-label small">
			<text x="430" y="215">Value layer</text>
			<text x="430" y="407">Value layer</text>
			<text x="430" y="599">Value layer</text>
		</g>

		<!-- ───── endpoint labels ───── -->
		<text class="label base-label" x="180" y="40">Token Prediction</text>
		<text class="label value-label" x="430" y="40">Value Prediction</text>
		<text class="label" x="180" y="786">Token Embedding</text>
		<text class="label" x="490" y="786">Last Reward</text>
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
		<li class="stopgrad">
			<svg class="swatch" viewBox="0 0 30 12" aria-hidden="true">
				<line class="wire" x1="2" y1="6" x2="28" y2="6" />
				<line class="slash" x1="11" y1="11" x2="15" y2="1" />
				<line class="slash" x1="16" y1="11" x2="20" y2="1" />
			</svg>
			<span>Stop-gradient</span>
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

	.sg line {
		stroke: var(--label);
		stroke-width: 2;
		stroke-linecap: round;
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

	.box.encode rect {
		fill: color-mix(in srgb, var(--value) 14%, var(--pico-background-color, Canvas));
		stroke: var(--value);
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

	.box-sublabel text {
		fill: #fff;
		opacity: 0.85;
		font-size: 11px;
		text-anchor: middle;
		dominant-baseline: middle;
	}

	.encode-label text {
		fill: var(--label);
		font-size: 11px;
		font-weight: 600;
		text-anchor: middle;
		dominant-baseline: middle;
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

	.legend li.stopgrad .swatch .wire {
		stroke: var(--fwd);
	}

	.legend li.stopgrad .swatch .slash {
		stroke: var(--label);
		stroke-linecap: round;
	}

	figcaption {
		max-width: 30rem;
		margin: 0.5rem auto 0;
		text-align: center;
		font-size: 0.85rem;
		color: var(--pico-muted-color);
	}
</style>
