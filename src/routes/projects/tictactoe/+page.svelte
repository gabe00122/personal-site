<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import { ModelWrapper } from './model';
	import { initalGameState, turn } from './game';
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
        game = initalGameState;
        prevousGame = initalGameState
        value = 0.0;
        preferneces = [0, 0, 0, 0, 0, 0, 0, 0, 0];

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
		prevousGame = game;
        preferneces = result.preferences;
        value = result.value;
		game = turn(game, result.action);

		aiCalculating = false;
        aiHasMoved = true;
	}

	let model: tf.GraphModel<string>;
	let modelWrapper: ModelWrapper;

	let loading = true;
	let gameStart = true;
    let showPreferences = false;
    let aiHasMoved = false;
    
    let playerWentFirst = true;
	let aiCalculating = false;
	
    let value = 0.0;
	
    let preferneces = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    let prevousGame = initalGameState;
    let game = initalGameState;

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

{#if gameStart}
	<div class="control-group">
		<button on:click={playAsX}>I want to go first</button>
		<button on:click={playAsO}>I want to go second</button>
	</div>
{:else if loading}
	<div class="control-group">Loading Model...</div>
{:else}
	<div class="board">
        {#each game.board as boardCell, index}
            {#if aiHasMoved && showPreferences && prevousGame.board[index] === undefined}
                <Cell cell={boardCell} on:click={() => onCellClick(index)} preference={preferneces[index]}></Cell>
            {:else}
                <Cell cell={boardCell} on:click={() => onCellClick(index)}></Cell>
            {/if}
        {/each}
	</div>

	{#if game.result !== 'ongoing'}
        <div style="text-align: center;">
            {#if game.result === 'draw'}
                It's a draw!
            {:else if playerWentFirst && game.activePlayer === 'O'}
                You won!
            {:else}
                You lost!
            {/if}
        </div>
		<div class="control-group" style="margin-bottom: 20px;">
			<button on:click={restart}>Play Again?</button>
		</div>
	{/if}

    <div class="control-group">Critic Value: {value.toFixed(3)}</div>
    <div class="control-group">
        <label>
            <input type="checkbox" name="english" bind:checked={showPreferences} />
            Show preferences
        </label>
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
</style>
