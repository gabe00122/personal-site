<script lang="ts">
    import { createEventDispatcher} from "svelte";
    import type { Cell } from "./game";

    const dispach = createEventDispatcher();

    function onClick() {
        dispach("click");
    }

    export let preference: number | undefined = undefined;
    export let cell: Cell;
</script>

<div class="cell" on:click={onClick} class:selectable={cell === undefined}>
    
    <div class="preference">
        {#if preference !== undefined}
            {preference.toLocaleString(undefined, { style: 'percent' })}
        {/if}
    </div>
    
    {#if cell === "X"}
        <svg class="icon" fill="#000000" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 460.775 460.775" xml:space="preserve">
            <path d="M285.08,230.397L456.218,59.27c6.076-6.077,6.076-15.911,0-21.986L423.511,4.565c-2.913-2.911-6.866-4.55-10.992-4.55  c-4.127,0-8.08,1.639-10.993,4.55l-171.138,171.14L59.25,4.565c-2.913-2.911-6.866-4.55-10.993-4.55  c-4.126,0-8.08,1.639-10.992,4.55L4.558,37.284c-6.077,6.075-6.077,15.909,0,21.986l171.138,171.128L4.575,401.505  c-6.074,6.077-6.074,15.911,0,21.986l32.709,32.719c2.911,2.911,6.865,4.55,10.992,4.55c4.127,0,8.08-1.639,10.994-4.55  l171.117-171.12l171.118,171.12c2.913,2.911,6.866,4.55,10.993,4.55c4.128,0,8.081-1.639,10.992-4.55l32.709-32.719  c6.074-6.075,6.074-15.909,0-21.986L285.08,230.397z"/>
        </svg>
    {:else if cell === "O"}
        <svg class="icon" fill="#000000" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1 1" xml:space="preserve">
            <circle stroke="black" fill="none" class="s-9KZLRJY08aZQ" cy="0.5" r="0.4" stroke-width="0.16" cx="0.5"></circle>
        </svg>
    {/if}
</div>

<style>
    .cell {
        background-color: lightgray;
    }

    .icon {
        width: 100%;
        padding: 0 25%;
    }

    .preference {
        color: black;
        height: 27%;
        text-align: center;
    }

    .selectable {
        cursor: pointer;
    }
</style>
