// Add `thebe` class to all python code blocks so Thebe can make them executable
window.addEventListener('load', () => {
  try {
    const codeSelectors = [
      'div.highlight pre > code.language-python',
      'pre > code.language-python',
      'div.highlight-python pre',
    ];

    const nodes = new Set();
    for (const sel of codeSelectors) {
      document.querySelectorAll(sel).forEach((el) => nodes.add(el));
    }

    nodes.forEach((codeEl) => {
      // add class to code element
      if (!codeEl.classList.contains('thebe')) codeEl.classList.add('thebe');

      // ensure parent pre has a container that Thebe recognizes
      const pre = codeEl.closest('pre') || codeEl.parentElement;
      if (pre && !pre.classList.contains('thebe')) pre.classList.add('thebe');
    });

    // If Thebe is loaded, re-bootstrap so newly-classed elements are picked up
    if (window.thebe && typeof window.thebe.bootstrap === 'function') {
      try {
        window.thebe.bootstrap(window.thebe_config || {});
      } catch (e) {
        console.warn('thebe.bootstrap failed on auto classing', e);
      }
    }
  } catch (err) {
    console.warn('thebe_autoclass failed', err);
  }
});
