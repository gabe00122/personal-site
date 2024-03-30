<header style="border-bottom: 1px solid #e8e8e8">
    <nav aria-label="Main" lang="en" class="mainnav container">
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
</header>

<slot></slot>

<style>
:global(body) {
    margin: 0;
    font-family: "Open Sans";
    font-size: 22px;

    background: #fdfdfd;
}

.container {
    width: 800px;
    margin: 0 auto;
}


.mainnav ul {
    display: flex;
    width: 100%;
    max-width: 500px;
    margin: 0 auto;
    padding: 0;
    list-style-type: none;
}

.mainnav li {
    flex-basis: 50%;
    flex-grow: 1;
    flex-shrink: 1;
}

.mainnav a {
    color: #424242;
    text-decoration: none;
    display: flex;
    align-items: center;
    justify-content: center;

    min-height: 44px;
}

.mainnav li:hover {
    text-decoration: underline;
}
</style>