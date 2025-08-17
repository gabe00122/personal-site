<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import { ModelWrapper } from './model';
	import { initialGameState, turn } from './game';
	import Cell from './cell.svelte';

	async function onCellClick(cellNumber: number) {
		const isValidAction = game.board[cellNumber] === undefined;

		if (!aiCalculating && game.result === 'ongoing' && isValidAction) {
			game = turn(game, cellNumber);

			if (game.result === 'ongoing') {
				await playAITurn();
			}
		}
	}

	function restart() {
		game = initialGameState;
		previousGame = initialGameState;
		value = 0.0;
		preferences = [0, 0, 0, 0, 0, 0, 0, 0, 0];

		gameStart = true;
		aiHasMoved = false;
	}

	function playAsX() {
		gameStart = false;
		playerWentFirst = true;
	}

	async function playAsO() {
		playerWentFirst = false;
		await playAITurn();
		gameStart = false;
	}

	async function playAITurn() {
		aiCalculating = true;

		const result = await modelWrapper.act(game);
		previousGame = game;
		preferences = result.preferences;
		value = result.value;
		game = turn(game, result.action);

		aiCalculating = false;
		aiHasMoved = true;
	}

	let model: tf.GraphModel<string>;
	let modelWrapper: ModelWrapper;

	let loading = $state(true);
	let gameStart = $state(true);
	let showPreferences = $state(false);
	let aiHasMoved = $state(false);

	let playerWentFirst = $state(true);
	let aiCalculating = false;

	let value = $state(0.0);

	let preferences = $state([0, 0, 0, 0, 0, 0, 0, 0, 0]);
	let previousGame = $state(initialGameState);
	let game = $state(initialGameState);

	onMount(async () => {
		if (!model) {
			model = await tf.loadGraphModel('/tictactoe/selfplay/model.json');
			modelWrapper = new ModelWrapper(model);

			// Warm up the model
			await modelWrapper.act(game);
			loading = false;
		}
	});
</script>

<svelte:head>
	<title>Reinforcement Learning tic tac toe demo</title>
	<meta
		name="description"
		content="Challenge yourself against an (nearly) unbeatable Tic Tac Toe AI trained with Reinforcement Learning."
	/>
</svelte:head>

{#if gameStart}
	<div class="control-group">
		<button onclick={playAsX}>I want to go first</button>
		<button onclick={playAsO}>I want to go second</button>
	</div>
{:else if loading}
	<div class="control-group">Loading Model...</div>
{:else}
	<div class="board">
		{#each game.board as boardCell, index}
			{#if aiHasMoved && showPreferences && previousGame.board[index] === undefined}
				<Cell cell={boardCell} on:click={() => onCellClick(index)} preference={preferences[index]}
				></Cell>
			{:else}
				<Cell cell={boardCell} on:click={() => onCellClick(index)}></Cell>
			{/if}
		{/each}
	</div>

	{#if game.result !== 'ongoing'}
		<div style="text-align: center;">
			{#if game.result === 'draw'}
				It's a draw!
			{:else if playerWentFirst === (game.activePlayer === 'O')}
				You won!
			{:else}
				You lost!
			{/if}
		</div>
		<div class="control-group" style="margin-bottom: 20px;">
			<button onclick={restart}>Play Again?</button>
		</div>
	{/if}

	<div class="control-group">Critic Value: {value.toFixed(3)}</div>
	<div class="control-group">
		<label>
			<input type="checkbox" name="english" bind:checked={showPreferences} />
			Show preferences
		</label>
	</div>
	<div class="centered-text">
		This model was train with self play in less than a minute on a RTX 3070<br >
		See training code on <a href="https://github.com/gabe00122/tictactoe-rl">github</a>
	</div>
{/if}

<style>
	.board {
		margin: 0 auto;
		width: 300px;
		height: 300px;
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		grid-template-rows: repeat(3, 1fr);
		gap: 2px;

		background-color: black;
	}

	.control-group {
		display: flex;
		justify-content: center;
		gap: 10px;
		margin: 10px;
	}

	.centered-text {
		text-align: center;
	}
</style>
