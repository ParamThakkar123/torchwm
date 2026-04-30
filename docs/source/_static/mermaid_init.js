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
            let txt = s.textContent || s.innerText || '';
            // Normalize common typographic characters that break Mermaid parsing
            // - smart quotes -> straight quotes
            // - en/em dashes -> hyphen
            // - arrow-like sequences (e.g. –> from copy/paste) -> -->
            // - replace Unicode minus with ASCII minus
            try {
              txt = txt.replace(/[\u201C\u201D]/g, '"'); // “ ”
              txt = txt.replace(/[\u2018\u2019\u201B]/g, "'"); // ‘ ’ ‛
              txt = txt.replace(/[\u2013\u2014\u2012]/g, '-'); // – — ‒ -> -
              // Convert en-dash/emdash followed by > into proper mermaid arrow
              txt = txt.replace(/[-\u2013\u2014]+>/g, '-->');
              // Convert right arrow glyphs to ASCII
              txt = txt.replace(/\u2192/g, '->');
              // Unicode minus to ASCII hyphen
              txt = txt.replace(/\u2212/g, '-');
            } catch (e) {
              // ignore normalization errors
            }
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
        // Render existing mermaid blocks. If MathJax is present, wait for
        // MathJax to finish typesetting so Mermaid doesn't try to render
        // content that MathJax will rewrite (avoids parse/type conflicts).
        try {
          const doRender = () => {
            if (typeof window.mermaid.run === 'function') {
              window.mermaid.run(document.querySelectorAll('.mermaid'));
            } else if (typeof window.mermaid.init === 'function') {
              window.mermaid.init(undefined, document.querySelectorAll('.mermaid'));
            }
          };

          if (window.MathJax && window.MathJax.startup && typeof window.MathJax.startup.typesetPromise === 'function') {
            // Wait for MathJax typesetting to complete, then render Mermaid
            window.MathJax.startup.typesetPromise().then(() => {
              try { doRender(); } catch (e) { console.warn('Mermaid render failed after MathJax', e); }
            }).catch(() => {
              // If MathJax errors, fall back to delayed render
              setTimeout(doRender, 100);
            });
          } else {
            // No MathJax or older API — render after a short delay
            setTimeout(doRender, 50);
          }
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
      const nodes = (typeof document.querySelectorAll === 'function') ? document.querySelectorAll('.mermaid') : null;
      if (nodes && nodes.length) {
        if (typeof window.mermaid.init === 'function') {
          window.mermaid.init(undefined, nodes);
        } else if (typeof window.mermaid.run === 'function') {
          // prefer calling run without args; fall back to nodes if needed
          try { window.mermaid.run(); } catch (err) { try { window.mermaid.run(nodes); } catch (e) { /* swallow */ } }
        }
      }
    } catch (e) {
      // swallow errors
    }
  });
  // Observe the body if available; otherwise observe the documentElement as a fallback
  const targetNode = document.body || document.documentElement || null;
  if (targetNode) {
    observer.observe(targetNode, { childList: true, subtree: true });
  } else {
    // If neither exists yet, attach when DOM is ready
    document.addEventListener('DOMContentLoaded', () => {
      const t = document.body || document.documentElement;
      if (t) observer.observe(t, { childList: true, subtree: true });
    });
  }
})();
