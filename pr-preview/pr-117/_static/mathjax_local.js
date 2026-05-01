/* Vendored MathJax runtime (trimmed/packaged) - wrapper loader
   We include a small bootstrap that loads the full MathJax runtime file
   saved alongside this file as mathjax_runtime.js. This avoids network
   dependency if hosting blocks CDN. */
// eslint-disable-next-line no-unused-vars
(function () {
  // If MathJax already present, skip
  if (window.MathJax) return;
  // Load the local runtime script (runtime is copied into _static; when
  // this loader is itself copied to _static, loading the runtime file by
  // basename is correct).
  var s = document.createElement('script');
  s.src = 'mathjax_runtime.js';
  s.async = true;
  document.head.appendChild(s);
})();
