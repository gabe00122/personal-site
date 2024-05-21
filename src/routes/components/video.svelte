<script lang="ts">
    export let url: string;
</script>

<div id="vid">Video Description</div>
<video autoplay loop muted aria-describedby="vid" class="centered">
    <source src={url} type="video/mp4">
    <!-- <track kind="captions"> -->
</video>


<style>
    .centered {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
</style>
