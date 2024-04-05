<script lang="ts">
    import { onMount } from "svelte";
    import * as tf from "@tensorflow/tfjs";
    import { initalGameState, turn } from "./game";
    import Cell from "./cell.svelte";

    let model: tf.GraphModel<string>;
    export let game = initalGameState;

    onMount(async () => {
        if (!model) {
            const board = tf.tensor([
                0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0,
                0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0,
                0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0,
            ]);
            console.log(tf.getBackend());

            const mask = tf.ones([9], "bool");
            model = await tf.loadGraphModel("/tictactoe/selfplay/model.json");

            const [value, actions] = model.predict({xs_0: board, xs_1: mask}) as [tf.Tensor, tf.Tensor];

            console.log(value.dataSync());
            console.log(actions.dataSync());
            console.log(tf.softmax(actions).dataSync());

        }
    });
</script>

<div class="board">
    {#each game.board as boardCell}
        <Cell></Cell>
    {/each}
</div>

<style>
    .board {
        display: grid;
        grid-template-columns: 50px 50px 50px;
        grid-template-rows: 50px 50px 50px; 
    }
</style>