// Small MathJax configuration that enables TeX/LaTeX support and sets options
// Configure MathJax before the runtime loads. Placed under _static so Sphinx
// copies it into the built site's _static directory and the page can load it.
window.MathJax = window.MathJax || {};
window.MathJax.tex = {
  inlineMath: [['$', '$'], ['\\(', '\\)']],
  displayMath: [['$$', '$$'], ['\\[', '\\]']],
  packages: {'[+]': ['ams']}
};
window.MathJax.options = {
  skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
  processHtmlClass: 'math',
};

// Ensure MathJax doesn't auto-start until the page is ready; we'll call
// typeset when the runtime has loaded via mathjax_init.js
window.MathJax.startup = window.MathJax.startup || {};
window.MathJax.startup.typeset = false;

// Expose a small flag so our init helper knows that config ran successfully
window.__MATHJAX_CONFIGURED__ = true;
