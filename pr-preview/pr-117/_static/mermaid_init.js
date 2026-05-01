// Robust Mermaid initialization: wait for mermaid to load, then render any
// <div class="mermaid"> elements. Re-run on DOM mutations.
(() => {
  const MAX_WAIT = 5000; // ms
  const INTERVAL = 100; // ms

  function tryInit() {
    if (window.mermaid) {
      try {
        // Convert any <pre class="mermaid"> blocks into <div class="mermaid"> for proper rendering
        try {
          const preBlocks = document.querySelectorAll('pre.mermaid');
          preBlocks.forEach((pre) => {
            let txt = pre.textContent || pre.innerText || '';
            // Normalize common typographic characters that break Mermaid parsing
            try {
              txt = txt.replace(/[\u201C\u201D]/g, '"'); // " "
              txt = txt.replace(/[\u2018\u2019\u201B]/g, "'"); // ' ' '
              txt = txt.replace(/[\u2013\u2014\u2012]/g, '-'); // - - -
              // Convert en-dash/emdash followed by > into proper mermaid arrow
              txt = txt.replace(/[-\u2013\u2014]+>/g, '-->');
              // Convert right arrow glyphs to ASCII
              txt = txt.replace(/\u2192/g, '->');
              // Unicode minus to ASCII hyphen  
              txt = txt.replace(/\u2212/g, '-');
              // Fix common syntax issues
              // Remove leading indentation that looks like code block formatting
              txt = txt.replace(/^[\t ]+/gm, '');
            } catch (e) {
              // ignore normalization errors
            }
            const d = document.createElement('div');
            d.className = 'mermaid';
            d.textContent = txt;
            pre.parentNode.replaceChild(d, pre);
          });
        } catch (e) {
          // ignore conversion errors  
        }
        // Also handle old-style script blocks
        try {
          document.querySelectorAll('script[type="text/vnd.mermaid"]').forEach((s) => {
            let txt = s.textContent || s.innerText || '';
            try {
              txt = txt.replace(/[\u201C\u201D]/g, '"');
              txt = txt.replace(/[\u2018\u2019\u201B]/g, "'");
              txt = txt.replace(/[\u2013\u2014\u2012]/g, '-');
              txt = txt.replace(/[-\u2013\u2014]+>/g, '-->');
              txt = txt.replace(/\u2192/g, '->');
              txt = txt.replace(/\u2212/g, '-');
              txt = txt.replace(/^[\t ]+/gm, '');
            } catch (e) {}
            const d = document.createElement('div');
            d.className = 'mermaid';
            d.textContent = txt;
            s.parentNode.replaceChild(d, s);
          });
        } catch (e) {}
        // Use loose security to allow HTML labels
        const dark = document.documentElement.dataset.theme === 'dark' || window.matchMedia('(prefers-color-scheme: dark)').matches;
        window.mermaid.initialize({ startOnLoad: false, securityLevel: 'loose', theme: dark ? 'dark' : 'default' });
        
        // Render with error handling
        try {
          const doRender = async () => {
            const mermaidDivs = document.querySelectorAll('.mermaid');
            if (!mermaidDivs.length) return;
            
            // Try parse first to catch syntax errors
            for (const div of mermaidDivs) {
              const txt = div.textContent || div.innerText;
              if (!txt || !txt.trim()) continue;
              
              try {
                const { svg } = await window.mermaid.render('mermaid-' + Math.random().toString(36).substr(2, 9), txt);
                div.innerHTML = svg;
                div.setAttribute('data-processed', 'true');
              } catch (parseErr) {
                console.warn('Mermaid parse error:', parseErr.message);
                div.innerHTML = '<p style="color:red;">Diagram error: ' + parseErr.message + '</p><pre style="background:#f5f5f5;padding:10px;">' + txt + '</pre>';
              }
            }
          };

          if (window.MathJax && window.MathJax.startup && typeof window.MathJax.startup.typesetPromise === 'function') {
            window.MathJax.startup.typesetPromise().then(() => {
              try { doRender(); } catch (e) { console.warn('Mermaid render failed', e); }
            }).catch(() => { setTimeout(doRender, 100); });
          } else {
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

  // Poll for mermaid availability
  let waited = 0;
  const poll = setInterval(() => {
    if (tryInit() || (waited > MAX_WAIT)) {
      clearInterval(poll);
    }
    waited += INTERVAL;
  }, INTERVAL);

  // Re-render on DOM mutations
  const observer = new MutationObserver(() => {
    if (!window.mermaid) return;
    try {
      const nodes = document.querySelectorAll('.mermaid:not([data-processed="true"])');
      if (nodes && nodes.length) {
        window.mermaid.run(nodes).catch(() => {});
      }
    } catch (e) {}
  });
  const targetNode = document.body || document.documentElement;
  if (targetNode) {
    observer.observe(targetNode, { childList: true, subtree: true });
  }
})();
