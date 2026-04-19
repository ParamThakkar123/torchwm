// Robust Mermaid initialization: wait for mermaid to load, then render any
// <div class="mermaid"> elements. Re-run on DOM mutations.
(() => {
  const MAX_WAIT = 5000; // ms
  const INTERVAL = 100; // ms

  function tryInit() {
    if (window.mermaid) {
      try {
        // Convert any <script type="text/vnd.mermaid"> blocks into
        // <div class="mermaid"> so mermaid.run/init can process them.
        try {
          document.querySelectorAll('script[type="text/vnd.mermaid"]').forEach((s) => {
            const txt = s.textContent || s.innerText || '';
            const d = document.createElement('div');
            d.className = 'mermaid';
            d.textContent = txt;
            s.parentNode.replaceChild(d, s);
          });
        } catch (e) {
          // ignore conversion errors
        }
        // Use loose security to allow HTML labels; don't auto-start (we'll run)
        // Configure theme based on document theme attribute (light/dark)
        const dark = document.documentElement.dataset.theme === 'dark' || window.matchMedia('(prefers-color-scheme: dark)').matches;
        window.mermaid.initialize({ startOnLoad: false, securityLevel: 'loose', theme: dark ? 'dark' : 'default' });
        // Render existing mermaid blocks after a short delay so MathJax (if
        // present) can process page elements first. This avoids conflicts
        // where MathJax rewrites nodes that Mermaid targets.
        try {
          const render = () => {
            if (typeof window.mermaid.run === 'function') {
              window.mermaid.run(document.querySelectorAll('.mermaid'));
            } else if (typeof window.mermaid.init === 'function') {
              window.mermaid.init(undefined, document.querySelectorAll('.mermaid'));
            }
          };
          // Give MathJax a moment to typeset (if present), then render mermaid.
          setTimeout(render, 50);
        } catch (e) {
          console.warn('Mermaid render failed', e);
        }
        return true;
      } catch (e) {
        console.warn('Mermaid initialize failed', e);
        return false;
      }
    }
    return false;
  }

  // Poll for mermaid availability for a short time
  let waited = 0;
  const poll = setInterval(() => {
    if (tryInit() || (waited > MAX_WAIT)) {
      clearInterval(poll);
    }
    waited += INTERVAL;
  }, INTERVAL);

  // Re-render mermaid blocks when new content appears (Thebe or dynamic sections)
  const observer = new MutationObserver((mutations) => {
    if (!window.mermaid) return;
    try {
      if (typeof window.mermaid.run === 'function') {
        window.mermaid.run(document.querySelectorAll('.mermaid'));
      } else if (typeof window.mermaid.init === 'function') {
        window.mermaid.init(undefined, document.querySelectorAll('.mermaid'));
      }
    } catch (e) {
      // swallow errors
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
})();
