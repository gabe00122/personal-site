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
			'projected into the value stream through SwiGLU value-encode blocks. The value gradient ' +
			'trains the encode blocks but stops at the ⫽ stop-gradient marks, so the policy and value ' +
			'gradients never cross.'
	}: Props = $props();
</script>

<figure class="arch-figure">
	<svg
		class="arch"
		viewBox="0 0 560 730"
		role="img"
		aria-label="A ladder value network running parallel to the Qwen3 base model. Grey arrows show
			the forward pass flowing up: the token embedding flows up through a stack of six base layers,
			each a frozen Qwen3 block with a LoRA adapter, to the token prediction. The token embedding,
			tapped through its own stop-gradient value-encode block, and the last reward together feed a
			parallel value stream of three smaller value layers ending in the value prediction. The
			output of every second base layer is tapped through a value-encode block into a value layer.
			Green arrows show the policy gradient flowing back down the base stack; red arrows show the
			value gradient flowing back down the value stack and into each value-encode block, where it
			stops at the stop-gradient mark before reaching the base model. The two backward streams
			never cross."
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
			<line x1="180" y1="680" x2="180" y2="612" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="560" x2="180" y2="524" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="472" x2="180" y2="436" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="384" x2="180" y2="348" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="296" x2="180" y2="260" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="208" x2="180" y2="172" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="120" x2="180" y2="54" marker-end="url(#arch-arrow)" />

			<!-- value residual stream (bottom segment rises from the input merge) -->
			<line x1="430" y1="642" x2="430" y2="564" marker-end="url(#arch-arrow)" />
			<line x1="430" y1="520" x2="430" y2="388" marker-end="url(#arch-arrow)" />
			<line x1="430" y1="344" x2="430" y2="212" marker-end="url(#arch-arrow)" />
			<line x1="430" y1="168" x2="430" y2="54" marker-end="url(#arch-arrow)" />

			<!-- taps: every 2nd base latent -> value encode -> value layer -->
			<line x1="180" y1="182" x2="376" y2="182" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="358" x2="376" y2="358" marker-end="url(#arch-arrow)" />
			<line x1="180" y1="534" x2="376" y2="534" marker-end="url(#arch-arrow)" />

			<!-- token embedding feeds the value net (and, separately, the policy above) -->
			<line x1="180" y1="642" x2="426" y2="642" marker-end="url(#arch-arrow)" />
			<!-- last reward feeds only the value net -->
			<path d="M490,684 L490,622 L434,622" fill="none" marker-end="url(#arch-arrow)" />
		</g>

		<!-- ───── policy backward pass (green, flows down the base stream) ───── -->
		<g class="bwd-policy">
			<line x1="160" y1="54" x2="160" y2="116" marker-end="url(#arch-arrow)" />
			<line x1="160" y1="168" x2="160" y2="204" marker-end="url(#arch-arrow)" />
			<line x1="160" y1="256" x2="160" y2="292" marker-end="url(#arch-arrow)" />
			<line x1="160" y1="344" x2="160" y2="380" marker-end="url(#arch-arrow)" />
			<line x1="160" y1="432" x2="160" y2="468" marker-end="url(#arch-arrow)" />
			<line x1="160" y1="520" x2="160" y2="556" marker-end="url(#arch-arrow)" />
		</g>

		<!-- ───── value backward pass (red, flows down the value stream) ───── -->
		<g class="bwd-value">
			<line x1="410" y1="54" x2="410" y2="164" marker-end="url(#arch-arrow)" />
			<line x1="410" y1="208" x2="410" y2="340" marker-end="url(#arch-arrow)" />
			<line x1="410" y1="384" x2="410" y2="516" marker-end="url(#arch-arrow)" />

			<!-- value gradient feeds into (but not through) each value-encode block -->
			<line x1="380" y1="194" x2="326" y2="194" marker-end="url(#arch-arrow)" />
			<line x1="380" y1="370" x2="326" y2="370" marker-end="url(#arch-arrow)" />
			<line x1="380" y1="546" x2="326" y2="546" marker-end="url(#arch-arrow)" />
			<path d="M410,560 L410,654 L326,654" fill="none" marker-end="url(#arch-arrow)" />
		</g>

		<!-- stop-gradient marks on the taps -->
		<g class="sg">
			<line x1="211" y1="190" x2="219" y2="174" />
			<line x1="220" y1="190" x2="228" y2="174" />
			<line x1="211" y1="366" x2="219" y2="350" />
			<line x1="220" y1="366" x2="228" y2="350" />
			<line x1="211" y1="542" x2="219" y2="526" />
			<line x1="220" y1="542" x2="228" y2="526" />
			<line x1="211" y1="650" x2="219" y2="634" />
			<line x1="220" y1="650" x2="228" y2="634" />
		</g>

		<!-- ───── base layers (green boxes) ───── -->
		<g class="box base">
			<rect x="105" y="120" width="150" height="48" rx="12" />
			<rect x="105" y="208" width="150" height="48" rx="12" />
			<rect x="105" y="296" width="150" height="48" rx="12" />
			<rect x="105" y="384" width="150" height="48" rx="12" />
			<rect x="105" y="472" width="150" height="48" rx="12" />
			<rect x="105" y="560" width="150" height="48" rx="12" />
		</g>

		<!-- ───── value encode blocks (on the taps) ───── -->
		<g class="box encode">
			<rect x="250" y="174" width="72" height="28" rx="7" />
			<rect x="250" y="350" width="72" height="28" rx="7" />
			<rect x="250" y="526" width="72" height="28" rx="7" />
			<rect x="250" y="634" width="72" height="28" rx="7" />
		</g>

		<!-- ───── value layers (red boxes) ───── -->
		<g class="box value">
			<rect x="380" y="168" width="100" height="40" rx="9" />
			<rect x="380" y="344" width="100" height="40" rx="9" />
			<rect x="380" y="520" width="100" height="40" rx="9" />
		</g>

		<!-- ───── endpoint labels ───── -->
		<text class="label base-label" x="180" y="40">Token Prediction</text>
		<text class="label value-label" x="430" y="40">Value Prediction</text>
		<text class="label" x="180" y="710">Token Embedding</text>
		<text class="label" x="490" y="710">Last Reward</text>
	</svg>

	<ul class="legend">
		<li class="box-base">
			<svg class="swatch" viewBox="0 0 30 12" aria-hidden="true">
				<rect x="4" y="0.75" width="22" height="10.5" rx="3" />
			</svg>
			<span>Base layer (frozen + LoRA)</span>
		</li>
		<li class="box-value">
			<svg class="swatch" viewBox="0 0 30 12" aria-hidden="true">
				<rect x="4" y="0.75" width="22" height="10.5" rx="3" />
			</svg>
			<span>Value layer</span>
		</li>
		<li class="box-encode">
			<svg class="swatch" viewBox="0 0 30 12" aria-hidden="true">
				<rect x="4" y="0.75" width="22" height="10.5" rx="3" />
			</svg>
			<span>Value encode</span>
		</li>
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
		--encode-fill: color-mix(in srgb, var(--value) 14%, var(--pico-background-color, Canvas));
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
	.bwd-value line,
	.bwd-value path {
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

	.bwd-value line,
	.bwd-value path {
		stroke: var(--value);
	}

	.sg line {
		stroke: var(--label);
		stroke-width: 2;
		stroke-linecap: round;
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
		fill: var(--encode-fill);
		stroke: var(--value);
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

	.swatch rect {
		stroke-width: 1;
	}

	.legend li.box-base .swatch rect {
		fill: var(--base);
		stroke: color-mix(in srgb, var(--base) 65%, #000);
	}

	.legend li.box-value .swatch rect {
		fill: var(--value);
		stroke: color-mix(in srgb, var(--value) 65%, #000);
	}

	.legend li.box-encode .swatch rect {
		fill: var(--encode-fill);
		stroke: var(--value);
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
