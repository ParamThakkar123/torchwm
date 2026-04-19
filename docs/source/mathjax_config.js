// Small MathJax configuration that enables TeX/LaTeX support and sets options
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    packages: {'[+]': ['ams']}
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
  }
};

// This file is referenced from conf.py so it is loaded before the MathJax runtime
