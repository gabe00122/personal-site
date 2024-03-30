<script lang="ts">
    import "@picocss/pico/css/pico.pumpkin.min.css";
</script>

<main class="container">
    <!-- <header style="border-bottom: 1px solid #e8e8e8">
        <nav aria-label="Main" lang="en">
            <ul>
                <li>
                    <a lang="en" href="/about"><span>About</span></a>
                </li>
                <li>
                    <a lang="en" href="/projects"><span>Projects</span></a>
                </li>
                <li>
                    <a lang="en" href="/blog"><span>Blog</span></a>
                </li>
                <li>
                    <a lang="en" href="/music"><span>Music</span></a>
                </li>
            </ul>
        </nav>
    </header> -->
    <nav>
        <ul>
          <li><strong>Gabriel Keith</strong></li>
        </ul>
        <ul>
          <li><a href="/about">About</a></li>
          <li><a href="/blog">Blog</a></li>
          <li><a href="/projects">Projects</a></li>
          <li><a href="/music">Music</a></li>
        </ul>
      </nav>

    <slot></slot>
</main>

