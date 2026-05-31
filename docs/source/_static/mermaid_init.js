// Reliable Mermaid bootstrap for Sphinx + MyST pages.
// sphinxcontrib-mermaid may emit either <div class="mermaid"> blocks or
// <script type="text/vnd.mermaid"> blocks depending on version/output mode.
// This script normalizes both forms and calls the Mermaid v10 API correctly.
(() => {
  const MAX_WAIT_MS = 5000;
  const POLL_MS = 100;

  function normalizeSource(source) {
    return (source || '')
      .replace(/[\u201C\u201D]/g, '"')
      .replace(/[\u2018\u2019\u201B]/g, "'")
      .replace(/[\u2013\u2014\u2012]/g, '-')
      .replace(/[-\u2013\u2014]+>/g, '-->')
      .replace(/\u2192/g, '->')
      .replace(/\u2212/g, '-');
  }

  function normalizeMermaidBlocks() {
    document.querySelectorAll('script[type="text/vnd.mermaid"]').forEach((script) => {
      const block = document.createElement('div');
      block.className = 'mermaid';
      block.textContent = normalizeSource(script.textContent || script.innerText || '');
      script.replaceWith(block);
    });

    document.querySelectorAll('.mermaid').forEach((block) => {
      if (!block.dataset.originalMermaid) {
        block.dataset.originalMermaid = normalizeSource(block.textContent || '');
        block.textContent = block.dataset.originalMermaid;
      }
    });
  }

  async function renderMermaid() {
    if (!window.mermaid) return false;

    normalizeMermaidBlocks();
    const nodes = Array.from(document.querySelectorAll('.mermaid:not([data-processed="true"])'));
    if (!nodes.length) return true;

    const dark = document.documentElement.dataset.theme === 'dark'
      || window.matchMedia('(prefers-color-scheme: dark)').matches;

    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'loose',
      theme: dark ? 'dark' : 'default',
    });

    try {
      if (typeof window.mermaid.run === 'function') {
        await window.mermaid.run({ nodes });
      } else if (typeof window.mermaid.init === 'function') {
        window.mermaid.init(undefined, nodes);
      }
      return true;
    } catch (error) {
      console.warn('Mermaid render failed', error);
      nodes.forEach((node) => {
        node.dataset.processed = 'false';
      });
      return false;
    }
  }

  function startPolling() {
    let waited = 0;
    const poll = window.setInterval(() => {
      renderMermaid().then((done) => {
        if (done || waited >= MAX_WAIT_MS) window.clearInterval(poll);
      });
      waited += POLL_MS;
    }, POLL_MS);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startPolling, { once: true });
  } else {
    startPolling();
  }

  const observer = new MutationObserver(() => {
    if (window.mermaid) renderMermaid();
  });

  const observe = () => {
    if (document.body) observer.observe(document.body, { childList: true, subtree: true });
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', observe, { once: true });
  } else {
    observe();
  }
})();
