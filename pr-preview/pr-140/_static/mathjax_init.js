// Ensure MathJax runs typesetting after the runtime loads.
(function () {
  const MAX_WAIT = 5000;
  const INTERVAL = 50;
  let waited = 0;

  function tryTypeset() {
    if (window.MathJax && window.MathJax.startup) {
      try {
        // If we set startup.typeset = false in config, we must call typesetPromise
        if (typeof window.MathJax.startup.typesetPromise === 'function') {
          window.MathJax.startup.typesetPromise()
            .catch((err) => console.warn('MathJax typeset failed', err));
        } else if (typeof window.MathJax.typeset === 'function') {
          // older API fallback
          try { window.MathJax.typeset(); } catch (e) { console.warn('MathJax.typeset failed', e); }
        }
        return true;
      } catch (e) {
        console.warn('MathJax init error', e);
        return false;
      }
    }
    return false;
  }

  const poll = setInterval(() => {
    if (tryTypeset() || (waited > MAX_WAIT)) {
      clearInterval(poll);
    }
    waited += INTERVAL;
  }, INTERVAL);

  // Re-run typeset when new content is added (e.g., Thebe or dynamic sections)
  try {
    const obs = new MutationObserver(() => {
      if (window.MathJax && typeof window.MathJax.startup?.typesetPromise === 'function') {
        window.MathJax.startup.typesetPromise().catch(() => {});
      }
    });
    obs.observe(document.body, { childList: true, subtree: true });
  } catch (e) {
    // ignore
  }
})();
