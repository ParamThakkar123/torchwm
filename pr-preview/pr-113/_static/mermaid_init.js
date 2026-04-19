// Initialize Mermaid after it loads
window.addEventListener('load', () => {
  if (window.mermaid) {
    try {
      window.mermaid.initialize({ startOnLoad: true, securityLevel: 'loose' });
    } catch (e) {
      // fail silently in docs build environments
      console.warn('Mermaid init failed', e);
    }
  }
});

// Also re-render mermaid blocks when new content is inserted (e.g. via Thebe)
const observer = new MutationObserver((mutations) => {
  if (window.mermaid) window.mermaid.init(undefined, document.querySelectorAll('.mermaid'));
});
observer.observe(document.body, { childList: true, subtree: true });
