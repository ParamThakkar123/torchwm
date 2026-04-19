// Small MathJax configuration that enables TeX/LaTeX support and sets options
// Configure MathJax before the runtime loads.
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

// Ensure MathJax doesn't auto-start until the page is ready; we'll let the
// runtime be loaded and then it will pick up the config.
window.MathJax.startup = window.MathJax.startup || {};
window.MathJax.startup.typeset = false;


// This file is referenced from conf.py so it is loaded before the MathJax runtime
