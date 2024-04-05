<script lang="ts">
    import { onMount } from "svelte";
    import * as tf from "@tensorflow/tfjs";
    import { ModelWrapper } from "./model";
    import { initalGameState, turn } from "./game";
    import Cell from "./cell.svelte";

    async function onCellClick(cellNumber: number) {
        game = turn(game, cellNumber);

        const result = await modelWrapper.act(game);
        
        value = result.value;
        game = turn(game, result.action);
    }

    function resetGame() {
        game = initalGameState;
    }

    let model: tf.GraphModel<string>;
    let modelWrapper: ModelWrapper;

    let value = 0.0;
    export let game = initalGameState;

    onMount(async () => {
        if (!model) {
            model = await tf.loadGraphModel("/tictactoe/selfplay/model.json");
            modelWrapper = new ModelWrapper(model);
        }
    });
</script>

<div class="board">
    {#each game.board as boardCell, index}
        <Cell cell={boardCell} on:click={() => onCellClick(index)}></Cell>
    {/each}
</div>

<div>The AI's value: {value}</div>

<button on:click={resetGame}>Reset</button>

<style>
    .board {
        margin: 0 auto;
        width: 300px;
        height: 300px;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        grid-template-rows: repeat(3, 1fr); 
    }
</style>